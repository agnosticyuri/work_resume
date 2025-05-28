# -*- coding: utf-8 -*-
"""

@author: jske
"""


import laspy
import numpy as np
from scipy.interpolate import griddata

# Load the LAS file
input_file = "C:/fastload/hrad50_merge.las"
las = laspy.read(input_file)

# Extract coordinates (X, Y, Z) and classification codes
points = np.vstack((las.x, las.y, las.z)).transpose()
classification = las.classification

# Separate ground points and points with classification code 50
ground_points = points[classification == 2]  # Code 2 represents ground points
ground_classifications = classification[classification == 2]

code_50_points = points[classification == 0]  # Code 50 for specific points
code_50_classifications = classification[classification == 0]

# Interpolate ground points for places under classification code 50
# Interpolate Z values for code_50_points' X and Y based on ground points
interpolated_z = griddata(ground_points[:, :2], ground_points[:, 2], code_50_points[:, :2], method='linear')

# Replace interpolated ground points where interpolation fails (NaN values)
default_ground_level = np.min(ground_points[:, 2])  # Lowest ground level
interpolated_z = np.where(np.isnan(interpolated_z), default_ground_level, interpolated_z)

# Create the new interpolated ground points
interpolated_ground_points = np.column_stack((code_50_points[:, 0], code_50_points[:, 1], interpolated_z))
interpolated_ground_classifications = np.full(interpolated_ground_points.shape[0], 2)  # Assign classification 2

# Combine ground points, interpolated ground points, and other points
non_ground_points = points[classification != 2]  # Points not classified as ground
non_ground_classifications = classification[classification != 2]

final_points = np.vstack((ground_points, interpolated_ground_points, non_ground_points))
final_classifications = np.concatenate((ground_classifications, interpolated_ground_classifications, non_ground_classifications))

#Save all points and their classifications to a new LAS file
header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
final_las = laspy.LasData(header)
final_las.x, final_las.y, final_las.z = final_points[:, 0], final_points[:, 1], final_points[:, 2]
final_las.classification = final_classifications  # Save classification codes

output_file = "C:/fastload/hrad50_mergeinterpol.las"
final_las.write(output_file)

print(f"Point cloud with interpolated ground and all classifications saved to: {output_file}")
