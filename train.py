import os
import torch
import torch.nn as nn
from data import OrthoViewDataset
from model import DinoViewSelector, pairwise_ranking_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    num_samples = 0
    for batch in tqdm(loader, total=len(loader)):
        batch['images'], batch['labels'] = batch['images'].to(device), batch['labels'].to(device)
        scores = model(batch['images'])
        loss = pairwise_ranking_loss(scores=scores, labels=batch['labels'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_samples += batch['images'].shape[0]

    return total_loss / num_samples

def validate(model, loader, device):
    model.eval()
    val_loss = 0.0

    num_samples = 0
    for batch in loader:
        with torch.no_grad():
            batch['images'], batch['labels'] = batch['images'].to(device), batch['labels'].to(device)
            scores = model(batch['images'])
            loss = pairwise_ranking_loss(scores=scores, labels=batch['labels'])

        val_loss += loss.item()
        num_samples += batch['images'].shape[0]

    return val_loss/num_samples

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parent_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views'
    train_set = OrthoViewDataset(f"{parent_dir}/train_.txt")
    val_set   = OrthoViewDataset(f"{parent_dir}/val_.txt")

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)#DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False)

    model = DinoViewSelector().to(device)

    optimizer = torch.optim.AdamW(
        model.scorer.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,       # aggressive decay
        patience=5,       # wait few epochs of no improvement
        min_lr=1e-6,
        verbose=True,
    )

    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:02d}] "
            f"loss={train_loss:.4f} "
            f"val_loss={val_loss:.3f} "
            f"lr={lr:.2e}")

        
        if epoch % 5==0:
            model_save_dir = f"{parent_dir}/model_saves_25.03.2026"
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{model_save_dir}/ortho_view_selector_{epoch}.pth")

if __name__ == "__main__":
    main()    