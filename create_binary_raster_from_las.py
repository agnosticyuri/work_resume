# -*- coding: utf-8 -*-
"""

@author: jske
"""


import laspy
import numpy as np
import gdal, osr

# Load the LAS file
las = laspy.read("D:/miloslav/merge_miloslavov.las")
#las = laspy.read("C:/martina/merge.las")
# Define target classification value (e.g., 2 for ground points)
target_classification = 6

# Extract points matching classification
classified_points = las.points[las.classification == target_classification]
x, y = classified_points.x, classified_points.y

# Define raster resolution and extent
xmin, xmax = np.min(x), np.max(x)
ymin, ymax = np.min(y), np.max(y)
resolution = 1  # Adjust resolution as needed

rows = int((ymax - ymin) / resolution)
cols = int((xmax - xmin) / resolution)

# Create empty binary raster
binary_raster = np.zeros((rows, cols), dtype=np.uint8)

# Convert points to raster indices with bounds checking
x_idx = np.clip(((x - xmin) / resolution).astype(int), 0, cols - 1)
y_idx = np.clip(((ymax - y) / resolution).astype(int), 0, rows - 1)

# Assign binary values safely
binary_raster[y_idx, x_idx] = 1  # Points in target classification -> 1

# Save raster to GeoTIFF using GDAL
driver = gdal.GetDriverByName("GTiff")
binary_dataset = driver.Create("D:/miloslav/miloslav.tif", cols, rows, 1, gdal.GDT_Byte)

binary_dataset.GetRasterBand(1).WriteArray(binary_raster)

# Set georeferencing info
geotransform = (xmin, resolution, 0, ymax, 0, -resolution)
binary_dataset.SetGeoTransform(geotransform)

# Define projection (assuming UTM or EPSG code)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)  # Replace with the correct EPSG code
binary_dataset.SetProjection(srs.ExportToWkt())

# Close dataset
binary_dataset = None

print("Binary GeoTIFF created successfully!")
