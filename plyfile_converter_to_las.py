# -*- coding: utf-8 -*-
"""
Created on Wed May 14 22:25:14 2025

@author: jske
"""


from plyfile import PlyData

# Load file using PlyData
ply_data = PlyData.read("C:/wuh_train/plateau_railway-20250513T213912Z-1-007/plateau_railway/train/Tibet-5.ply")

# Print available properties (columns)
print("Available columns:", ply_data.elements[0].properties)
print("Available columns:", ply_data.elements[0].properties[4])
print("Available columns:", ply_data.elements[0])
vertex_properties = ply_data["vertex"].properties
print("Available columns:", vertex_properties)

import numpy as np

# Extract point coordinates
points = np.vstack([ply_data["vertex"]["x"], 
                    ply_data["vertex"]["y"], 
                    ply_data["vertex"]["z"]]).T

# Extract color if available
if {"red", "green", "blue"}.issubset(vertex_properties):
    colors = np.vstack([ply_data["vertex"]["red"], 
                        ply_data["vertex"]["green"], 
                        ply_data["vertex"]["blue"]]).T
else:
    colors = None

# Extract normals if available
if {"nx", "ny", "nz"}.issubset(vertex_properties):
    normals = np.vstack([ply_data["vertex"]["nx"], 
                         ply_data["vertex"]["ny"], 
                         ply_data["vertex"]["nz"]]).T
else:
    normals = None

intensity_ply = np.array(ply_data["vertex"]["intensity"], dtype=np.float32)
classification_ply = np.array(ply_data["vertex"]["class"], dtype=np.uint8)

print("Extracted", points.shape[0], "points!")


import laspy

# Create LAS header (point_format=3 for classification support)
header = laspy.LasHeader(point_format=3, version="1.2")
las = laspy.LasData(header)

# Assign classification values
las.classification = classification_ply
las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
las.intensity=intensity_ply
# Save LAS file
las.write("C:/wuh_train/output.las")

print("PLY file converted to LAS with class labels!")

ply_data=None
las=None