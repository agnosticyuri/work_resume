# -*- coding: utf-8 -*-
"""
@author: jske
"""


import geopandas as gpd
import laspy
import numpy as np
from shapely.geometry import Point, Polygon
import sys
# Load the shapefile containing polygons
shapefile_path = "D:/miloslav/miloslav_hullregion.shp"
polygon_gdf = gpd.read_file(shapefile_path)

# Extract polygon vertices
polygon_list = [Polygon(poly.exterior.coords) for poly in polygon_gdf.geometry]
all_poly=int(len(polygon_list))
# Load the LiDAR LAS file
las_file_path = "D:/miloslav/merge_miloslavov.las"
las = laspy.read(las_file_path)
target_classification = 6

# Extract points matching classification
classified_points = las.points[las.classification == target_classification]
# Convert LAS points to NumPy arrays
x = np.array(classified_points.x)
y = np.array(classified_points.y)
z = np.array(classified_points.z)
points = np.column_stack((x, y, z))
classified_points=None
las=None
# Boolean mask for points inside polygons
clipped_mask = np.zeros(len(points), dtype=bool)
# Iterate through polygons and mark points inside
counter=0
for polygon in polygon_list:
    inside_mask = np.array([polygon.contains(Point(px, py)) for px, py in points[:, :2]])
    clipped_mask |= inside_mask# Accumulate points inside polygons
    counter=+1
    remain= int(all_poly-counter)
    sys.stdout.write(f"\rPolygons remaining:{remain}")
    sys.stdout.flush()
# Extract clipped points
clipped_points = points[clipped_mask]

print(f"Clipped {len(clipped_points)} points.")

# Save clipped points separately
#np.savetxt("clipped_points.txt", clipped_points, delimiter=",", header="X,Y,Z", comments="")
