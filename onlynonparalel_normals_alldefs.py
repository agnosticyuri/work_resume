# -*- coding: utf-8 -*-
"""

@author: jske
"""


import numpy as np
import open3d as o3d
import laspy

def load_las_point_cloud(file_path, classification_code):
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


def filter_ground_parallel_normals(pcd, min_angle=50, max_angle=85, min_height=5.0):
    """
    Filters flat ground points while keeping flat roofs based on height.
    Keeps points where normal forms an angle in range [min_angle, max_angle],
    but allows flat surfaces above a certain height threshold (roofs).
    """
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    
    # Compute normal angle with ground
    angles = np.degrees(np.arccos(np.abs(normals[:, 2])))  

    # Filter points: Remove only if they are flat *and* close to ground level
    mask = (angles > min_angle) | (points[:, 2] > min_height) | (angles > max_angle)
    
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)

    return filtered_pcd
def filter_ground_parallel_normals(pcd, min_angle=5, max_angle=85):
    """
    Filters points whose normal vectors are oriented nearly flat but keeps sloped surfaces.
    Keeps points where the normal forms an angle between min_angle and max_angle degrees.
    """
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    # Compute normal angle with ground plane (Z-axis)
    angles = np.degrees(np.arccos(np.abs(normals[:, 2])))  

    # Filter points with normals that are *too parallel* to the ground
    mask = (angles > min_angle) & (angles < max_angle)
    
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)

    return filtered_pcd



def save_filtered_points_to_las(file_path, point_cloud, normals):
    """Save filtered point cloud and normals to a LAS file."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    las = laspy.LasData(header)

    las.x = point_cloud[:, 0]
    las.y = point_cloud[:, 1]
    las.z = point_cloud[:, 2]

    las.add_extra_dim(laspy.ExtraBytesParams(name="normal_x", type="float32"))
    las.add_extra_dim(laspy.ExtraBytesParams(name="normal_y", type="float32"))
    las.add_extra_dim(laspy.ExtraBytesParams(name="normal_z", type="float32"))

    las.normal_x = normals[:, 0]
    las.normal_y = normals[:, 1]
    las.normal_z = normals[:, 2]
    
    las.write(file_path)
    print(f"Saved filtered normals to {file_path}")

if __name__ == "__main__":
    classification_code = 6
    file_path = "C:/mesto/merge.las"#"D:/budovy_clip/LAS/merge.las"
    output_file_path = "C:/mesto/mergeopen3d8550.las"#"D:/budovy_clip/LAS/martina/open3d_normals.las"

    point_cloud = load_las_point_cloud(file_path, classification_code)
    print(f"Loaded {point_cloud.shape[0]} points.")

    radius = 1.0

    pcd = estimate_normals_open3d(point_cloud, radius)
    print("Surface normals estimated using Open3D.")

    filtered_pcd = filter_ground_parallel_normals(pcd)
    filtered_normals = np.asarray(filtered_pcd.normals)
    filtered_points = np.asarray(filtered_pcd.points)

    save_filtered_points_to_las(output_file_path, filtered_points, filtered_normals)

    # Visualize filtered normals
    filtered_pcd.paint_uniform_color([1, 0, 0])  # Color filtered points red
    o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)
