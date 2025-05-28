# -*- coding: utf-8 -*-
"""

@author: jske
"""


import numpy as np
import open3d as o3d
import laspy

def load_point_cloud(file_path, classification_code):
    """Load and filter point cloud data by a specific classification code from a LAS file."""
    las = laspy.read(file_path)
    mask = las.classification == classification_code
    x = las.x[mask]
    y = las.y[mask]
    z = las.z[mask]
    return np.vstack((x, y, z)).T

def estimate_normals_open3d(point_cloud, radius):
    """Estimate surface normals using Open3D's built-in normal estimation."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius))

    return pcd

def color_points_by_z(pcd):
    """Color point cloud based on Z values using a gradient."""
    points = np.asarray(pcd.points)
    z_values = points[:, 2]  # Extract Z values
    z_min, z_max = np.min(z_values), np.max(z_values)  # Find range

    # Normalize Z values to [0,1] for color mapping
    normalized_z = (z_values - z_min) / (z_max - z_min)
    
    # Apply color gradient: From blue (low) to red (high)
    colors = np.column_stack((normalized_z, np.zeros_like(normalized_z), 1 - normalized_z))
    pcd.colors = o3d.utility.Vector3dVector(colors)

def visualize_normals(pcd):
    """Render the point cloud with Open3D's built-in normal visualization."""
    pcd.paint_uniform_color([0, 0.8, 0])  # Color points green for better visibility
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud with Normals", point_show_normal=True)

if __name__ == "__main__":
    absolute_value = True
    classification_code = 6
    file_path = "C:/mesto/merge.las"#"D:/budovy_clip/LAS/merge.las"
    output_file_path = "C:/mesto/mergeopen3d.las"#"D:/budovy_clip/LAS/martina/open3d_normals.las"

    point_cloud = load_point_cloud(file_path, classification_code)
    print(f"Loaded {point_cloud.shape[0]} points.")

    radius = 1.0

    pcd = estimate_normals_open3d(point_cloud, radius)
    
    print("Surface normals estimated using Open3D.")

    # Apply color mapping based on Z values
    color_points_by_z(pcd)

    visualize_normals(pcd)
