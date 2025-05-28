# -*- coding: utf-8 -*-
"""

@author: jske
"""


import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

# Load and preprocess point cloud
pcd = o3d.io.read_point_cloud("point_cloud.ply")  # Load your point cloud file
points = np.asarray(pcd.points)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.05, min_samples=10)
labels = dbscan.fit_predict(points)

# Region-growing refinement
def region_growing(points, labels, threshold=0.02):
    refined_labels = labels.copy()
    unique_labels = set(labels)

    for cluster in unique_labels:
        if cluster == -1:  # Ignore noise
            continue

        cluster_points = points[labels == cluster]
        seed_idx = np.random.choice(len(cluster_points))
        seed_point = cluster_points[seed_idx]

        for i, point in enumerate(cluster_points):
            if np.linalg.norm(seed_point - point) < threshold:
                refined_labels[i] = cluster

    return refined_labels

# Apply region-growing
refined_labels = region_growing(points, labels)

# Visualize results
pcd.colors = o3d.utility.Vector3dVector(np.random.rand(len(refined_labels), 3))  # Assign random colors
o3d.visualization.draw_geometries([pcd])

