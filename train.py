import os
import torch
import torch.nn as nn
from data import OrthoViewDataset
from model import DinoViewSelector, pairwise_ranking_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(model, loader, val_loader, optimizer, device, val_freq=10):
    model.train()
    total_loss = 0.0
    last_val_loss = None  # Use None to track if it has run yet

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for i, batch in pbar:
        # --- Training Step ---
        batch['images'] = batch['images'].to(device)
        batch['labels'] = batch['labels'].to(device)
        
        scores = model(batch['images'])
        loss = pairwise_ranking_loss(scores=scores, labels=batch['labels'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_train_loss = total_loss / (i + 1)

        # --- Intermediate Validation ---
        # Trigger validation every val_freq batches
        # i + 1 is used so that if val_freq is 10, it runs on batch 10, 20, 30...
        if (i + 1) % val_freq == 0:
            last_val_loss = validate(model, val_loader, device)
            model.train() # Back to train mode

        # --- Update Progress Bar ---
        # Construct the display dictionary
        display_stats = {
            'loss': f"{loss.item():.6E}",
            'avg_train': f"{avg_train_loss:.6E}"
        }
        
        # Add val loss to display only after the first one is calculated
        if last_val_loss is not None:
            display_stats['val_loss'] = f"{last_val_loss:.6E}"
        else:
            display_stats['val_loss'] = "waiting..."

        pbar.set_postfix(display_stats)

    return avg_train_loss

def validate(model, loader, device):
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch['images'] = batch['images'].to(device)
            batch['labels'] = batch['labels'].to(device)
            
            scores = model(batch['images'])
            loss = pairwise_ranking_loss(scores=scores, labels=batch['labels'])
            
            val_loss += loss.item()
            num_batches += 1

    return val_loss / num_batches if num_batches > 0 else 0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parent_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM'
    model_save_dir = f"{parent_dir}/model_saves_27.04.2026"
    os.makedirs(model_save_dir, exist_ok=True)

    # --- Hyperparameters ---
    MAX_EPOCHS = 50
    PATIENCE_LIMIT = 3
    TOP_K_TO_KEEP = 3
    VAL_FREQ = 1  # Check validation loss every 5 batches

    train_set = OrthoViewDataset(f"{parent_dir}/train.log")
    val_set   = OrthoViewDataset(f"{parent_dir}/val.log")
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=16, shuffle=False)

    model = DinoViewSelector(num_views=110).to(device)
    optimizer = torch.optim.AdamW(model.scorer.parameters(), lr=1e-4, weight_decay=1e-4)

    # --- Tracking ---
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    checkpoint_history = [] 

    for epoch in range(MAX_EPOCHS):
        # Pass val_loader and val_freq to the training function
        train_loss = train_one_epoch(model, train_loader, val_loader, optimizer, device, val_freq=VAL_FREQ)
        
        # Final validation at end of epoch for checkpointing
        val_loss = validate(model, val_loader, device)

        print(f"\n[Epoch {epoch:02d} Results] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # --- Top-K Saving Logic ---
        checkpoint_path = f"{model_save_dir}/ep{epoch}_val{val_loss:.4f}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_history.append((val_loss, checkpoint_path))
        checkpoint_history.sort(key=lambda x: x[0])

        if len(checkpoint_history) > TOP_K_TO_KEEP:
            _, worst_path = checkpoint_history.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)

        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print(f"★ New Best Score!")
        else:
            epochs_without_improvement += 1
            print(f"Patience: {epochs_without_improvement}/{PATIENCE_LIMIT}")

        if epochs_without_improvement >= PATIENCE_LIMIT:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()