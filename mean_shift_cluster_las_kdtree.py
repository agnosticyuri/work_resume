# -*- coding: utf-8 -*-
"""
@author: jske
"""


import laspy
import numpy as np
from sklearn.cluster import MeanShift
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# Load point cloud data
las_file = laspy.read("D:/lasfile.las")
points = np.vstack((las_file.x, las_file.y, las_file.z)).T

# Build KD-Tree for efficient neighbor searching
tree = KDTree(points)

# Adaptive bandwidth estimation based on point density
bandwidth = np.median([tree.query(points[i], k=5)[0].mean() for i in range(len(points))])

# Fast Mean Shift clustering using precomputed neighborhoods
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
labels = mean_shift.fit_predict(points)

