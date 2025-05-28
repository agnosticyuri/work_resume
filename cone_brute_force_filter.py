# -*- coding: utf-8 -*-
"""

@author: jske
"""


import laspy
import numpy as np
import math

def is_inside_cone(px, py, pz, peak_x, peak_y, peak_z, k, max_height):
    """
    Check if point (px, py, pz) is inside a limited-height cone.

    Returns:
        True if the point is inside the cone, False otherwise.
    """
    if pz <= peak_z or pz > (peak_z + max_height):  # Limit height
        return False  # Must be within height range

    r = math.sqrt((px - peak_x)**2 + (py - peak_y)**2)  # Radial distance
    return r <= k * (pz - peak_z)  # Check cone boundary

def process_point_cloud(las_file_path, k=0.1, max_height=10):
    """
    Load a LAS point cloud, determine peak points considering limited cone height.
    
    Returns:
        NumPy array of peak points that have no other points in their cone.
    """
    las = laspy.read(las_file_path)
    classifications = las.classification  
    selected_class = 6  
    filtered_points = las.points[classifications == selected_class]
    points = np.vstack((filtered_points.x, filtered_points.y, filtered_points.z)).T
    peak_points = []
    
    for peak in points:
        peak_x, peak_y, peak_z = peak
        has_points_inside = False
        
        for p in points:
            if np.array_equal(p, peak):  
                continue
            
            if is_inside_cone(p[0], p[1], p[2], peak_x, peak_y, peak_z, k, max_height):
                has_points_inside = True
                break
        
        if not has_points_inside:
            peak_points.append(peak)

    return np.array(peak_points), las  # Convert list to NumPy array

def save_point_cloud(peak_points, las, output_path="C:/peak_allhg1k01.las"):
    """
    Save peak points to a new LAS file.

    """
    header = laspy.LasHeader(point_format=6, version="1.4")
    new_las = laspy.LasData(header)
    new_las.x = peak_points[:, 0]
    new_las.y = peak_points[:, 1]
    new_las.z = peak_points[:, 2]

    new_las.write(output_path)

# Start code
las_file_path = "C:/merge.las"
max_height =float(1.0) #float(input("Enter max cone height: "))  # Get user input

peak_points, las = process_point_cloud(las_file_path, k=0.5, max_height=max_height)
if peak_points.size > 0:
    save_point_cloud(peak_points, las)
    print(f"Number of peak points: {peak_points.shape[0]}")
else:
    print("No points found")
