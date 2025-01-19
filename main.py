import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# Define Fiber Analysis Model
class FiberAnalysisModel(nn.Module):
    def __init__(self, num_fiber_classes=3, num_damage_classes=2):
        super(FiberAnalysisModel, self).__init__()
        
        # Shared feature extraction
        self.layer1 = ResidualBlock(3, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fiber type classifier
        self.fc_fiber = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_fiber_classes)
        )
        
        # Damage type classifier
        self.fc_damage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_damage_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        fiber_out = self.fc_fiber(x)
        damage_out = self.fc_damage(x)
        return fiber_out, damage_out

# Prepare Data
def prepare_data(data_dir, batch_size=32):
    """
    Prepare data loaders with advanced augmentation.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAdjustSharpness(2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset.classes

# Training Function
def train_model(model, train_loader, optimizer, criterion_fiber, criterion_damage, epochs=10, device="cuda"):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            fiber_labels = labels[:, 0].to(device)  # Fiber labels
            damage_labels = labels[:, 1].to(device)  # Damage labels
            
            optimizer.zero_grad()
            fiber_preds, damage_preds = model(images)
            loss_fiber = criterion_fiber(fiber_preds, fiber_labels)
            loss_damage = criterion_damage(damage_preds, damage_labels)
            loss = loss_fiber + loss_damage
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# Evaluation Function
def evaluate_model(model, test_loader, device="cuda"):
    model.eval()
    fiber_preds_all, fiber_labels_all = [], []
    damage_preds_all, damage_labels_all = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            fiber_labels = labels[:, 0].to(device)
            damage_labels = labels[:, 1].to(device)
            fiber_preds, damage_preds = model(images)
            fiber_preds_all.extend(torch.argmax(fiber_preds, dim=1).cpu().numpy())
            fiber_labels_all.extend(fiber_labels.cpu().numpy())
            damage_preds_all.extend(torch.argmax(damage_preds, dim=1).cpu().numpy())
            damage_labels_all.extend(damage_labels.cpu().numpy())
    
    print("\nFiber Type Classification Report:")
    print(classification_report(fiber_labels_all, fiber_preds_all))
    print("\nDamage Type Classification Report:")
    print(classification_report(damage_labels_all, damage_preds_all))

# Main
if __name__ == "__main__":
    # Paths to your dataset
    data_dir = "/path/to/fiber_dataset"
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 15
    
    # Prepare data
    train_loader, test_loader, classes = prepare_data(data_dir, batch_size)
    print(f"Classes: {classes}")
    
    # Initialize model
    model = FiberAnalysisModel(num_fiber_classes=3, num_damage_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_fiber = nn.CrossEntropyLoss()
    criterion_damage = nn.CrossEntropyLoss()
    
    # Train model
    train_model(model, train_loader, optimizer, criterion_fiber, criterion_damage, epochs)
    
    # Evaluate model
    evaluate_model(model, test_loader)
