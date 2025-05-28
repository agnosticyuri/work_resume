# -*- coding: utf-8 -*-
"""

@author: jske
"""


import numpy as np
import laspy
from sklearn.decomposition import PCA

def calculate_normals(points, k=3):
    normals = np.zeros(points.shape, dtype=np.float32)
    for i in range(points.shape[0]):
        # Get the k nearest neighbors
        distances = np.sum((points - points[i])**2, axis=1)
        neighbors_idx = np.argsort(distances)[1:k+1]
        neighbors = points[neighbors_idx]
        
        # Perform PCA
        pca = PCA(n_components=3)
        pca.fit(neighbors)
        
        # Normal is the eigenvector corresponding to the smallest eigenvalue
        normals[i] = pca.components_[-1]
    
    return normals

def filter_points_by_normals(points, normals):
    # Filter points based on normals' x, y, and z components being positive or negative
    mask = (normals[:, 0] > 0) & (normals[:, 1] > 0) & (normals[:, 2] > 0)
    return points[mask]

def main():
    input_file = 'input.las'
    output_file = 'filtered_output.las'

    with laspy.open(input_file) as in_file:
        las = in_file.read()
        points = np.vstack((las.X, las.Y, las.Z)).transpose()

        normals = calculate_normals(points)
        filtered_points = filter_points_by_normals(points, normals)

        # Create a new LAS header with the correct number of points
        new_header = laspy.LasHeader(point_format=las.point_format, version=las.version)
        new_header.point_count = filtered_points.shape[0]
        
        # Create a new LAS file and write filtered points
        new_las = laspy.LasData(new_header)
        new_las.X = filtered_points[:, 0]
        new_las.Y = filtered_points[:, 1]
        new_las.Z = filtered_points[:, 2]

        new_las.write(output_file)

if __name__ == "__main__":
    main()
