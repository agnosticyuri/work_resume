# -*- coding: utf-8 -*-
"""

@author: jske
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNetPlusPlus, self).__init__()

        # Layer 1: Feature Extraction
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)  # Input: xyz coordinates
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)

        # Layer 2: Feature Aggregation
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.maxpool(x).squeeze(-1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Initiate model
if __name__ == "__main__":
    # Simulate a batch of point cloud data: batch_size, num_points, dimensions
    point_cloud = torch.rand(8, 3, 1024)

    # Initialize and test the PointNet++ model
    model = PointNetPlusPlus(num_classes=40)  # Assuming 40 classes for ModelNet40 dataset
    output = model(point_cloud)
    print(output.shape)  # Expected shape: (8, 40)
