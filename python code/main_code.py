import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Define a custom dataset class
class ThermalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]
        img_path = f"{self.root_dir}/{img_name}"
        image = Image.open(img_path).convert("RGB")  # Assuming thermal images are grayscale

        labels = torch.tensor(self.data_frame.iloc[idx, 1:], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset and split into train and validation sets
csv_path = "/home/labadmin/R7A_group11/Poultry disease detection.v6i.multiclass/train/_classes.csv"
root_dir = "/home/labadmin/R7A_group11/Poultry disease detection.v6i.multiclass/tran"
dataset = ThermalDataset(csv_file=csv_path, root_dir=root_dir, transform=transform)

# Split the dataset into training and validation sets
train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load pre-trained VGG-16 model
vgg16 = models.vgg16(pretrained=True)
vgg16.to(device)

# Modify the classifier to output 5 classes
vgg16.classifier[6] = nn.Linear(4096, 5)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(vgg16.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    vgg16.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    vgg16.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg16(inputs)
            val_loss = criterion(outputs, labels)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Extract features using the pre-trained model
vgg16.eval()
features = []
with torch.no_grad():
    for inputs, _ in valid_loader:
        inputs = inputs.to(device)
        outputs = vgg16.features(inputs)
        features.append(outputs)

# Concatenate features
features = torch.cat(features)
print("Extracted features shape:", features.shape)
