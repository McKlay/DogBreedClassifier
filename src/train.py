import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.data_loader import get_data_loaders
from src.model import get_model

def train_model(data_dir="data/train",
                num_classes=20,
                model_name="efficientnet_b0",
                epochs=10,
                batch_size=32,
                image_size=224,
                lr=0.0001,
                save_path="models/efficientnet_b0.pth",
                device=None):
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")

    # Load data
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    # Load model
    model = get_model(num_classes=num_classes, model_name=model_name)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        val_epoch_loss = val_loss / total_val
        val_epoch_acc = val_corrects.double() / total_val
        print(f"Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")

        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Best model saved with acc: {best_val_acc:.4f}")

    print(f"\n[INFO] Training complete. Best Val Accuracy: {best_val_acc:.4f}")

    return model, class_names
