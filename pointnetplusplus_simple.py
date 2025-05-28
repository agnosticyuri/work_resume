# -*- coding: utf-8 -*-
"""


@author: jske
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import laspy
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
# Load .las data with optional features
def load_las_data(file_path, use_extra_features=False):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    labels = las.classification  # Extract classification labels

    if use_extra_features:
        normals = np.vstack((las.nx, las.ny, las.nz)).transpose()  # Extract normals (if available)
        curvature = las.curvature  # Extract curvature (if available)
        sphericity = las.sphericity  # Extract sphericity (if available)
        features = np.hstack((normals, curvature[:, np.newaxis], sphericity[:, np.newaxis]))  # Combine features
        return points, features, labels
    else:
        return points, labels

# Custom Dataset for point clouds
class PointCloudDataset(Dataset):
    def __init__(self, points, labels=None, normalize=True):
        self.points = points
        self.labels = labels
        self.normalize = normalize

        # Normalize point cloud
        if self.normalize:
            self.points = self.points / np.max(self.points, axis=0)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point = self.points[idx]
        label = self.labels[idx] if self.labels is not None else None

        # Reshape point cloud for Conv1D input
        point = torch.tensor(point, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        if label is not None:
            label = torch.tensor(label, dtype=torch.long)
        return (point, label) if label is not None else point

# Define PointNet++
class PointNetPlusPlus(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PointNetPlusPlus, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)  # Global feature aggregation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training
def train_model(model, train_loader, num_classes, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model

# Classification Function
def classify_point_cloud(model, point_cloud, normalize=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if normalize:
        point_cloud = point_cloud / np.max(point_cloud, axis=0)

    point_cloud = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
    predictions = model(point_cloud)
    return torch.argmax(predictions, dim=1).cpu().item()

# Initiate training
if __name__ == "__main__":
    # Paths to .las files
    train_file = "path_to_train.las"
    test_file = "path_to_test.las"

    # Load training data
    train_points, train_labels = load_las_data(train_file, use_extra_features=False)

    # Create Dataset and DataLoader
    train_dataset = PointCloudDataset(train_points, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define model and train
    input_dim = 3  # XYZ coordinates
    num_classes = len(np.unique(train_labels))
    model = PointNetPlusPlus(input_dim=input_dim, num_classes=num_classes)
    model = train_model(model, train_loader, num_classes)

    # Load unclassified point cloud
    test_points, _ = load_las_data(test_file, use_extra_features=False)
    predicted_label = classify_point_cloud(model, test_points)
    print("Predicted Label:", predicted_label)
