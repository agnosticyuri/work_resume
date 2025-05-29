# -*- coding: utf-8 -*-
"""
Optimized version of the LAS processing script
@author: jske
"""

import laspy
import numpy as np
from scipy.spatial import KDTree
from joblib import Parallel, delayed  # For parallelization
import time
start_time = time.time()
# Load LAS point cloud file
file_path ="C:/clipping_2/LAS/gku_hrad_4050607080.las" #"D:/budovy_clip/LAS/merge_40.las"
las = laspy.read(file_path)

# Extract coordinates & classification codes
points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)  # Use float32 to optimize memory
classification = las.classification

# Filter only points with class code 40 (buildings)
building_mask = classification == 40
filtered_points = points[building_mask]

# Build KD-Tree using X, Y, and Z for fast nearest-neighbor search
tree = KDTree(filtered_points)

# Define proximity threshold and height limits
threshold = 0.2  # Horizontal proximity limit
low_threshold = 0.1  # Minimum height above reference point
high_threshold = 15.0  # Maximum height above reference point

# Function to check isolation efficiently
def check_isolation(i):
    x0, y0, z0 = filtered_points[i]

    # Find points within height range efficiently
    height_mask = (filtered_points[:, 2] > z0 + low_threshold) & (filtered_points[:, 2] <= z0 + high_threshold)
    nearby_points = filtered_points[height_mask]

    # If no nearby points exist, classify as isolated
    if nearby_points.shape[0] == 0:
        return filtered_points[i]

    # Fast distance computation
    distances = np.linalg.norm(nearby_points[:, :2] - [x0, y0], axis=1)

    return filtered_points[i] if not np.any(distances < threshold) else None

# Parallelized processing for speed boost
results = Parallel(n_jobs=-1)(delayed(check_isolation)(i) for i in range(len(filtered_points)))
isolated_points = [p for p in results if p is not None]

# Convert to NumPy array
isolated_points_array = np.array(isolated_points)

# Save result if there are isolated points
if isolated_points_array.size > 0:
    output_path = "D:/budovy_clip/LAS/martina2/hrad40.las"
    las_new = laspy.create()
    las_new.x = isolated_points_array[:, 0]
    las_new.y = isolated_points_array[:, 1]
    las_new.z = isolated_points_array[:, 2]
    las_new.write(output_path)
    
    print(f"Total isolated building points: {len(isolated_points_array)} (Optimized processing)")
end_time = time.time()
difference= end_time - start_time
print(f"Time to process: {difference} seconds")