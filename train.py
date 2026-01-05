# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import ViTConfig
from src.ls_vit import LSViTForAction
from src.dataset import HMDB51Dataset, collate_fn
from tqdm import tqdm

def train():
    # Cấu hình
    config = ViTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8 # Tùy chỉnh theo VRAM
    NUM_FRAMES = 16
    FRAME_STRIDE = 2
    EPOCHS = 10
    LR = 1e-4

    # Dataset & Dataloader
    train_ds = HMDB51Dataset(root="data/hmdb51", split="train", num_frames=NUM_FRAMES, frame_stride=FRAME_STRIDE)
    val_ds = HMDB51Dataset(root="data/hmdb51", split="val", num_frames=NUM_FRAMES, frame_stride=FRAME_STRIDE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Model
    model = LSViTForAction(config, num_classes=51).to(device)
    
    # Loss & Optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler() # Mixed precision

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for videos, labels in pbar:
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * videos.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss/total_train:.4f}, Train Acc: {train_correct/total_train:.4f} | "
              f"Val Loss: {val_loss/total_val:.4f}, Val Acc: {val_correct/total_val:.4f}")

if __name__ == "__main__":
    train()