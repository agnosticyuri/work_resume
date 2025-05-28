# -*- coding: utf-8 -*-
"""

@author: jske
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import laspy
import os

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

# Filter points of interest based on target classification labels
def filter_points_of_interest(points, labels, target_classes):
    filtered_points = []
    filtered_labels = []
    for i in range(len(points)):
        if labels[i] in target_classes:
            filtered_points.append(points[i])
            filtered_labels.append(labels[i])
    return np.array(filtered_points), np.array(filtered_labels)

# Define the PointCNN++ model
class PointCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PointCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Train the model with creating checkpoints
def train_model(model, train_data, train_labels, epochs=10, batch_size=32, learning_rate=0.001, checkpoint_path="checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()

    start_epoch = 0

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]

            batch_data = torch.tensor(batch_data, dtype=torch.float32).unsqueeze(1).permute(0, 2, 1).to(device)  # Reshape for Conv1D
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # Save a checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Save the final model
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

# Load the saved model
def load_model(save_path, input_dim, num_classes):
    model = PointCNN(input_dim, num_classes)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    return model

# Classify new point cloud
def classify_point_cloud(model, point_cloud, features=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if features is not None:
        combined_data = np.hstack((point_cloud, features))  # Combine point cloud and features
    else:
        combined_data = point_cloud

    combined_data = torch.tensor(combined_data, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)  # Reshape for Conv1D
    predictions = model(combined_data)
    return torch.argmax(predictions, dim=1).cpu().item()  # Move to CPU for output

# Example usage
if __name__ == "__main__":
    # Paths
    train_file = "C:/fastload/hrad50_mergeinterpolgrfilt2.las"
    test_file = "C:/fastload/hrad50.las"
    model_save_path = "C:/fastload/pointcnn_model.pth"
    checkpoint_path = "C:/fastload/checkpoint.pth"

    # Feature usage flag
    use_extra_features = False  # Set to True if you want to use normals, curvature, and sphericity
    target_classes = [1, 2]  # Specify target classification labels (e.g., [1, 2] for vegetation/buildings)

    # Load data
    if use_extra_features:
        train_points, train_features, train_labels = load_las_data(train_file, use_extra_features=True)
        test_points, test_features, test_labels = load_las_data(test_file, use_extra_features=True)
    else:
        train_points, train_labels = load_las_data(train_file, use_extra_features=False)
        test_points, test_labels = load_las_data(test_file, use_extra_features=False)

    # Filter points of interest
    train_points, train_labels = filter_points_of_interest(train_points, train_labels, target_classes)
    test_points, test_labels = filter_points_of_interest(test_points, test_labels, target_classes)

    # Preprocess data (e.g., normalize)
    train_points = train_points / np.max(train_points, axis=0)
    test_points = test_points / np.max(test_points, axis=0)

    # Convert to PyTorch tensors
    train_points = torch.tensor(train_points, dtype=torch.float32).unsqueeze(1).permute(0, 2, 1)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    test_points = torch.tensor(test_points, dtype=torch.float32).unsqueeze(1).permute(0, 2, 1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create and train model
    input_dim = train_points.shape[1]  # Only use coordinates
    num_classes = len(np.unique(train_labels))
    model = PointCNN(input_dim, num_classes)

    # Train model with checkpointing
    train_model(model, train_points, train_labels, checkpoint_path=checkpoint_path)

    # Save the final model
    save_model(model, model_save_path)

    # Load the model and classify new data
    loaded_model = load_model(model_save_path, input_dim, num_classes)
    new_points, new_labels = load_las_data("C:/fastload/hrad50.las", use_extra_features=False)
    new_points, new_labels = filter_points_of_interest(new_points, new_labels, target_classes)
    new_points = new_points / np.max(new_points, axis=0)
    new_points = torch.tensor(new_points, dtype=torch.float32).unsqueeze(1).permute(0, 2, 1)
    prediction = classify_point_cloud(loaded_model, new_points)
    print("Prediction:", prediction)
