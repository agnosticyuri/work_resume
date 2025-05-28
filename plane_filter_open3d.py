# -*- coding: utf-8 -*-
"""

@author: jske
"""


import laspy
import open3d as o3d
import numpy as np

# Load the LAS file using laspy
input_file = "C:/fastload/hrad50_nonoise110_2.las"
las = laspy.read(input_file)

# Extract point coordinates (X, Y, Z) into a NumPy array
points = np.vstack((las.x, las.y, las.z)).transpose()

# Convert the loaded points to an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Plane segmentation using Open3D
distance_threshold = 0.01  # Maximum allowable distance to the plane
ransac_n = 3               # Number of points to sample for RANSAC
num_iterations = 1000      # Number of iterations for RANSAC

plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                         ransac_n=ransac_n,
                                         num_iterations=num_iterations)

[a, b, c, d] = plane_model
print(f"Detected Plane Equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# Extract inlier and outlier points
plane = pcd.select_by_index(inliers)
remaining_cloud = pcd.select_by_index(inliers, invert=True)

# Save the segmented plane points back to a LAS file
filtered_points = np.asarray(plane.points)
header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
filtered_las = laspy.LasData(header)
filtered_las.x, filtered_las.y, filtered_las.z = filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2]

output_file = "C:/fastload/hrad50_nonoise110_2plane.las"
filtered_las.write(output_file)

print(f"Segmented plane saved to: {output_file}")

# Visualize the result
plane.paint_uniform_color([1, 0, 0])  # Red color for the plane
remaining_cloud.paint_uniform_color([0, 1, 0])  # Green color for the remaining points
o3d.visualization.draw_geometries([plane, remaining_cloud])
