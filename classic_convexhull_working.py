# -*- coding: utf-8 -*-
"""

@author: jske
"""
import numpy as np
import gdal, osr
import skimage.morphology as morph
from shapely.geometry import Polygon
import fiona
from fiona.crs import from_epsg

# Open the raster with GDAL
dataset = gdal.Open("D:/merge_binary_raster6.tif")
band = dataset.GetRasterBand(1)

# Read raster data into a NumPy array
binary_raster = band.ReadAsArray()

# Compute convex hull
convex_hull = morph.convex_hull_object(binary_raster)

# Extract boundary coordinates
coords = np.column_stack(np.where(convex_hull))  # Pixel locations

# Get geotransformation info
geotransform = dataset.GetGeoTransform()
xmin, pixel_size, _, ymax, _, pixel_size_y = geotransform

# Convert raster indices to lat. lon.
real_coords = [(xmin + col * pixel_size, ymax + row * pixel_size_y) for row, col in coords]

# Create a polygon from convex hull points
convex_polygon = Polygon(real_coords).convex_hull  # Ensure valid convex shape

# Define the spatial reference mostly 4326
srs = osr.SpatialReference()
srs.ImportFromWkt(dataset.GetProjection())

# Save as shapefile (.shp)
schema = {"geometry": "Polygon", "properties": {"id": "int"}}

with fiona.open("D:/merge_binary_hull.shp", "w", driver="ESRI Shapefile", crs=srs.ExportToProj4(), schema=schema) as shp:
    shp.write({"geometry": convex_polygon.__geo_interface__, "properties": {"id": 1}})

print("Convex hull saved as shapefile successfully!")

# Close dataset
dataset = None
