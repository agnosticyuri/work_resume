# -*- coding: utf-8 -*-
"""

@author: jske
"""


import laspy
import numpy as np

def merge_point_clouds(file1, file2, output_file):
    # Load the first point cloud file
    with laspy.open(file1) as file1_las:
        file1_points = file1_las.read()
        file1_coords = np.vstack((file1_points.x, file1_points.y, file1_points.z)).T
        file1_classes = file1_points.classification

    # Load the second point cloud file
    with laspy.open(file2) as file2_las:
        file2_points = file2_las.read()
        file2_coords = np.vstack((file2_points.x, file2_points.y, file2_points.z)).T
        file2_classes = file2_points.classification

    # Filter points in file2 that are already in file1
    filtered_coords, filtered_classes = zip(*[
        (point, cls) for point, cls in zip(file2_coords, file2_classes) 
        if not np.any(np.all(file1_coords == point, axis=1))])
    filtered_coords = np.array(filtered_coords)
    filtered_classes = np.array(filtered_classes)

    # Merge coordinates and classification codes
    merged_coords = np.vstack((file1_coords, filtered_coords))
    merged_classes = np.concatenate((file1_classes, filtered_classes))

    # Create a new LAS file and write merged points and classifications
    header = laspy.LasHeader(point_format=file1_points.point_format, version=file1_points.header.version)
    merged_points = laspy.LasData(header)
    merged_points.x, merged_points.y, merged_points.z = merged_coords[:, 0], merged_coords[:, 1], merged_coords[:, 2]
    merged_points.classification = merged_classes

    with laspy.open(output_file, mode="w", header=header) as output_las:
        output_las.write(merged_points)

# File paths
file1 = "C:/fastload/hrad50_mergeinterpolgrfilt2.las"
file2 = "C:/fastload/hrad50.las"
output_file = "C:/fastload/hrad50_mergefortrain.las"

# Merge point clouds
merge_point_clouds(file1, file2, output_file)
