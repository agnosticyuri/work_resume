# -*- coding: utf-8 -*-
"""
@author: jske
"""


import laspy
import geopandas as gpd
from shapely.geometry import Point

# Load the shapefile
shapefile_path = "D:/hullregion.shp"
gdf = gpd.read_file(shapefile_path)

# Load the LAS file
las_file_path = "D:/merged.las"
las = laspy.read(las_file_path)
target_classification = 6

# Extract points matching classification
classified_points = las.points[las.classification == target_classification]
x, y, z = classified_points.x, classified_points.y, classified_points.z
# Convert LAS points to GeoDataFrame
#points = [Point(x, y) for x, y in zip(las.x, las.y)]
#points = [Point(x, y, z) for x, y, z in zip(x,y,z)]
points = [Point(x,y) for x,y in zip(x,y)]
las_gdf = gpd.GeoDataFrame(geometry=points)

# Iterate over polygons and clip points
for index, row in gdf.iterrows():
    polygon = row['geometry']
    clipped_points = las_gdf[las_gdf.geometry.within(polygon)]
    
    # Perform a function on the clipped ROI
    point_count = clipped_points.shape[0]
    print(f"Polygon {index} contains {point_count} points")

    
