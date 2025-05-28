# -*- coding: utf-8 -*-
"""

@author: jske
"""


import laspy
import numpy as np
from scipy.spatial import KDTree

def is_point_in_inverted_cone_vectorized(base_points, radius, height, all_points):
    """Efficiently check which points lie inside upside-down cones using NumPy."""
    x0, y0, z0 = base_points.T  # Extract base point coordinates
    x, y, z = all_points.T      # Extract all point coordinates

    # Height constraint: Points must be **below** the base point
    height_mask = (z0 >= z) & (z >= z0 - height)

    # Radial constraint: Check scaled distance condition
    radial_distance = (x - x0)**2 + (y - y0)**2
    height_ratio = ((z0 - z) / height)**2  # Inverted cone ratio

    radial_mask = (radial_distance / radius**2) <= height_ratio

    return height_mask & radial_mask  # Combined logical mask

# Load LiDAR data from .las file
las_file = laspy.read("C:/merge.las")

# Extract point cloud coordinates & classification
filtered_points = np.vstack((las_file.x, las_file.y, las_file.z)).T
classifications = las_file.classification  

# Define classification filter (e.g., ground = 2)
selected_class = 6  
filtered_points = filtered_points[classifications == selected_class]  # Apply filtering

# Define cone parameters
radius = 0.25  # Base radius in meters
height = 5.0  # Cone height in meters

# Use KDTree for fast spatial searching
tree = KDTree(filtered_points)

# Process points in bulk
isolated_mask = np.ones(len(filtered_points), dtype=bool)  # Assume all points are isolated initially

for i, base_point in enumerate(filtered_points):
    # Query neighbors **within initial base radius** (efficient spatial lookup)
    neighbors_idx = tree.query_ball_point(base_point[:2], r=radius)  

    # Extract neighbor points
    neighbors = filtered_points[neighbors_idx]

    # Apply inverted cone check **only** on nearby points (avoids redundant checking)
    inside_mask = is_point_in_inverted_cone_vectorized(np.array([base_point]), radius, height, neighbors)

    if np.any(inside_mask):  
        isolated_mask[i] = False  # Mark point as **not isolated**

# Select only isolated points
isolated_points = filtered_points[isolated_mask]

# Save isolated points to a new LAS file
header = las_file.header
new_las = laspy.create(file_version=header.version, point_format=header.point_format)

new_las.x = isolated_points[:, 0]
new_las.y = isolated_points[:, 1]
new_las.z = isolated_points[:, 2]

new_las.write("C:/merge_coconerad.las")

print(f"Saved {len(isolated_points)} isolated points to filtered_inverted_lidar_data.las 🚀")
