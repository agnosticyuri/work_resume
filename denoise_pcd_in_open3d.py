# -*- coding: utf-8 -*-
"""

@author: jske
"""


import open3d as o3d
import laspy
import numpy as np

# Load the LAS file
input_file = "C:/fastload/hrad50_nonoise110.las"#"C:/fastload/hrad50.las"
las = laspy.read(input_file)

# Extract point coordinates (X, Y, Z)
points = np.vstack((las.x, las.y, las.z)).transpose()

# Convert points to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Apply radius outlier removal
radius = 1.0  # Radius for neighbour search
min_neighbors = 10  # Minimum number of neighbors to keep a point

filtered_pcd, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)

# Save the filtered point cloud back to a LAS file
filtered_points = np.asarray(filtered_pcd.points)
header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
filtered_las = laspy.LasData(header)
filtered_las.x, filtered_las.y, filtered_las.z = filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2]

output_file = "C:/fastload/hrad50_nonoise110_2.las"
filtered_las.write(output_file)

print(f"Filtered point cloud saved to: {output_file}")
