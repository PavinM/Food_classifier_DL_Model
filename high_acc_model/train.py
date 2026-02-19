import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Configuration
# >90% accuracy goal requires a good model and data augmentation.
# EfficientNet-B0 is a great balance.
MODEL_NAME = "effective_food_model.pth"
DATA_DIR = "../data" # Relative path to existing data
BATCH_SIZE = 32
EPOCHS = 3 # Reduced for testing
LEARNING_RATE = 0.0001 # Lower LR for fine-tuning

# Device
device = torch.device("cpu")
print(f"Using device: {device}")

# Transforms (Augmentation is Key for Accuracy)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)), # Optimized for speed
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train():
    # Data Loading
    train_dir = os.path.join(DATA_DIR, "training")
    val_dir = os.path.join(DATA_DIR, "validation")
    
    if not os.path.exists(train_dir):
        print(f"Error: Data directory not found at {train_dir}")
        return

    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # Save class names for the app
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")

    # Model Setup
    print("Initializing EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Unfreeze all layers for maximum accuracy (fine-tuning)
    for param in model.parameters():
        param.requires_grad = True
        
    # Replace Head
    in_features = model.classifier[1].in_features
    # Add a more robust classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2), # Dropout to prevent overfitting
        nn.Linear(512, num_classes)
    )
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Use a lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Scheduler to reduce LR when plateauing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    print(f"Starting training for {EPOCHS} epochs...")
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        train_acc = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_NAME)
            print(f"New best model saved with accuracy: {val_acc:.2f}%")
            
    print("Training Complete.")

if __name__ == "__main__":
    train()
