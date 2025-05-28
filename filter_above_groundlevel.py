# -*- coding: utf-8 -*-
"""

@author: jske
"""


import laspy
import numpy as np
from scipy.spatial import KDTree

# Load the LAS file
input_file = "C:/fastload/hrad50_mergeinterpol.las"
las = laspy.read(input_file)

# Extract coordinates (X, Y, Z) and classification codes
points = np.vstack((las.x, las.y, las.z)).transpose()
classification = las.classification

# Step 1: Split ground points by classification code = 2
ground_points = points[classification == 2]  # Code 2 represents ground points
non_ground_points = points[classification != 2]  # All other non-ground points

# Step 2: Build KDTree using only X, Y coordinates of ground points
tree = KDTree(ground_points[:, :2])

# Step 3: Calculate heights of non-ground points above ground points
# Query nearest ground point in X, Y plane for each non-ground point
distances, indices = tree.query(non_ground_points[:, :2])  # X, Y of non-ground points
ground_z_values = ground_points[indices, 2]  # Z-values of nearest ground points
heights_above_ground = non_ground_points[:, 2] - ground_z_values  # Height above ground

# Step 4: Filter non-ground points above the height threshold
height_threshold = 2.0  # Threshold in meters
filtered_points = non_ground_points[heights_above_ground > height_threshold]

# Step 5: Save filtered points to a new LAS file
header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
filtered_las = laspy.LasData(header)
filtered_las.x, filtered_las.y, filtered_las.z = filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2]

output_file = "C:/fastload/hrad50_mergeinterpolgrfilt2.las"
filtered_las.write(output_file)

print(f"Filtered points above {height_threshold} meters saved to: {output_file}")
