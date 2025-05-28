# -*- coding: utf-8 -*-
"""

@author: jske
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import laspy
# Create a synthetic point cloud
#np.random.seed(0)
#points = np.random.rand(100, 2)  # 100 random 2D points
las_file_path ="D:/miloslav/merge_miloslavov.las" #"C:/martina/merge.las"
las = laspy.read(las_file_path)
#classifications = las.classification  
selected_class = 6  
filtered_points = las.points[las.classification == selected_class]
#points = np.vstack((filtered_points.x, filtered_points.y)).T
del las
print(f"Number of points: {len(filtered_points)}")
point_count=len(str(int(filtered_points)))
mil=int(1000000)
computed=point_count/mil

print(point_count)
#dividend = 10
#divisor = 4
remainder = point_count % mil
#div= point_count/mil
if remainder != 0:
    print("The result is a float.")
    computed_k=computed+1
else:
    print("The result is an integer.")
    computed_k=computed
"""
# Apply k-means clustering
k = 2  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(points)

# Extract cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print(f"{np.unique(np.asarray(labels))}")
# Plot results
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
plt.legend()
plt.title(f'K-Means Clustering with {k} Clusters')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
"""