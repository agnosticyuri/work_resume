# -*- coding: utf-8 -*-
"""

@author: jske
"""


import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import laspy
import time

def load_point_cloud(file_path):
    """
    Load point cloud data from a LAS file.
    """
    las = laspy.read(file_path)
    x = las.x
    y = las.y
    z = las.z
    return np.vstack((x, y, z)).T  # Combine x, y, z into a single array

def classify_point(eigenvalues):
    """
    Classify point based on eigenvalues.
    """
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
    e1, e2, e3 = sorted_eigenvalues

    if e1 > 2 * e2 and e2 > 2 * e3:  # Strong linear feature
        return "linear"
    elif e1 > e2 > e3 and e2 > 2 * e3:  # Planar
        return "planar"
    elif np.isclose(e1, e2) and np.isclose(e2, e3):  # Spherical
        return "spherical"
    elif e2 > 2 * e3:  # Tilted planar or curvy
        return "tilted_planar"
    else:  # Default category if no strong feature detected
        return "curvy"

def compute_pca_neighborhood(point_cloud, neighborhoods):
    """
    Compute PCA for neighborhoods of each point and classify geometry.
    """
    classifications = []
    pca = PCA(n_components=3)

    for neighbors in neighborhoods:
        if neighbors.shape[0] < 3:  # Skip if there are not enough points for PCA estimation
            classifications.append("undetermined")
            continue
        pca.fit(neighbors)
        eigenvalues = pca.explained_variance_
        classification = classify_point(eigenvalues)
        classifications.append(classification)
    
    return classifications

def find_neighborhoods(point_cloud, radius):# radius = 1.0
    """
    Find neighborhoods using KDTree (radius-based neighborhood search).
    """
    tree = KDTree(point_cloud)
    neighborhoods = []
    
    for point in point_cloud:
        indices = tree.query_ball_point(point, radius)
        neighbors = point_cloud[indices]
        neighborhoods.append(neighbors)
    
    return neighborhoods

if __name__ == "__main__":
    startime=time.time()
    # Path to your LAS file
    las_file_path = "C:/merge.las"
    
    # Load point cloud data
    point_cloud = load_point_cloud(las_file_path)
    print(f"Point Cloud Loaded: {point_cloud.shape[0]} points.")
    
    # Define radius for neighborhood search
    radius = 2.5

    # Find neighborhoods
    neighborhoods = find_neighborhoods(point_cloud, radius)
    print(f"Computed neighborhoods for {len(point_cloud)} points.")
    
    # Compute PCA and classify points
    classifications = compute_pca_neighborhood(point_cloud, neighborhoods)
    print("Point Classifications:")
    unique_classes, counts = np.unique(classifications, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        print(f"{cls}: {count} points")
    endtime=time.time() 
    timed=endtime-startime
    print("time",timed)