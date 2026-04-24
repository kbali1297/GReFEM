import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from data import OrthoViewDataset
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# DINO backbone (frozen)
# -----------------------------
class DinoEncoder(nn.Module):
    def __init__(self, model_name="dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.embed_dim = self.backbone.embed_dim

    def forward(self, x):
        """
        x: (B*V, 3, H, W)
        returns: (B*V, D)
        """
        feats = self.backbone(x)
        return feats


# -----------------------------
# View Selection Network
# -----------------------------
class ViewScoringNet(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x):
        """
        x: (B, V, D)
        returns: (B, V)
        """
        attn_out, _ = self.attn(x, x, x)   # self-attention
        scores = self.mlp(attn_out).squeeze(-1)
        return scores


# -----------------------------
# Full model
# -----------------------------
class DinoViewSelector(nn.Module):
    def __init__(self, num_views):
        super().__init__()
        self.encoder = DinoEncoder()
        self.scorer = ViewScoringNet(
            embed_dim=self.encoder.embed_dim
        )

        self.positional_embeddings = nn.Parameter(torch.randn(num_views, self.encoder.embed_dim) * 0.02)

    def forward(self, images):
        """
        images: (B, V, 3, H, W)
        """
        B, V, C, H, W = images.shape
        images = images.view(B * V, C, H, W)

        feats = self.encoder(images)
        feats = feats.view(B, V, -1)

        #Add positional embeddings to the view features here if desired (e.g., learnable or fixed positional embeddings based on view index)
        feats = feats + self.positional_embeddings.unsqueeze(0)  # (1, V, D) -> broadcast to (B, V, D)
        scores = self.scorer(feats)
        return scores


# -----------------------------
# Loss Function
# -----------------------------
def pairwise_ranking_loss(scores, labels, margin=1.0):
    """
    scores: (B, V)
    labels: (B, V)  binary or integer
    """
    loss = scores.sum() * 0.0
    count = 0

    for b in range(scores.shape[0]):
        pos = scores[b][labels[b] > 0]
        neg = scores[b][labels[b] == 0]

        if len(pos) == 0 or len(neg) == 0:
            continue

        diff = margin - (pos[:, None] - neg[None, :])
        loss += torch.clamp(diff, min=0).mean()
        count += 1

    if count == 0:
        return scores.sum() * 0.0

    return loss / count

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/model_saves/ortho_view_selector_20.pth'
    view_selector_model = DinoViewSelector().to(device)
    view_selector_model.load_state_dict(torch.load(model_path, map_location=device))
    
    view_selector_model.eval()
    val_dataset = OrthoViewDataset(dataset_list_path='/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/val.txt')

    val_loader = DataLoader(
                    val_dataset,
                    batch_size=2,        # or >1 if your samples fit in memory
                    shuffle=False,       # NEVER shuffle validation
                    num_workers=4,       # adjust for your node
                    pin_memory=True      # important when using GPU
                    )
    
    ## Run inference
    for batch in val_loader:
        batch['images'], batch['labels'] = batch['images'].to(device), batch['labels'].to(device)
        scores = view_selector_model(batch['images'])
        print('inspecting scores')