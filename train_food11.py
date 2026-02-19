import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Device Configuration
device = torch.device("cpu")
print("Using device:", device)

# Transforms with Augmentation for Training
train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Transforms for Validation (No Augmentation)
val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

data_dir = "./data"
train_dir = os.path.join(data_dir, "training")
val_dir = os.path.join(data_dir, "validation")

print(f"Loading data from: {train_dir}")

try:
    # Use ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    
    print("Dataset loaded successfully.")
    
    class_names = train_dataset.classes
    print("Classes:", class_names)
    print("Number of classes:", len(class_names))

    # Full Dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model (Same architecture, just num_classes=11)
    class FoodCNN(nn.Module):
        def __init__(self, num_classes=11):
            super(FoodCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

            self.pool = nn.MaxPool2d(2, 2)

            self.fc1 = nn.Linear(128 * 16 * 16, 512)
            self.fc2 = nn.Linear(512, num_classes)

            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))

            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)

            return x

    model = FoodCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 15
    print(f"Starting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        train_loss = running_loss/len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

    # Save
    torch.save(model.state_dict(), "food_model.pth")
    print("Model saved to food_model.pth")
    
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print("Class names saved to class_names.txt")

except Exception as e:
    print(f"Error loading/training: {e}")
