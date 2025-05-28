# -*- coding: utf-8 -*-
"""
@author: jske

"""


import numpy as np
import gdal
import skimage.morphology as morph
import skimage.measure as measure
from shapely.geometry import Polygon
import fiona
from fiona.crs import from_epsg

# Open the binary raster with GDAL
dataset = gdal.Open("D:/binary_image.tif")#D:/miloslav/miloslav.tif D:/miloslav/merge_binary_raster605.tif
band = dataset.GetRasterBand(1)

# Read raster data into a NumPy array
binary_raster = band.ReadAsArray()

# Label connected regions
labeled_regions, num_regions = measure.label(binary_raster, return_num=True, connectivity=2)

# Get geotransformation info
geotransform = dataset.GetGeoTransform()
xmin, pixel_size, _, ymax, _, pixel_size_y = geotransform

# Define shapefile schema
schema = {"geometry": "Polygon", "properties": {"id": "int"}}

# Create a shapefile to store convex hulls
with fiona.open("D:/hullregion.shp", "w", driver="ESRI Shapefile", crs=from_epsg(4326), schema=schema) as shp:#4326 32633
    for region_id in range(1, num_regions + 1):  # Ignore background (0)
        # Extract pixels belonging to this region
        region_mask = labeled_regions == region_id
        coords = np.column_stack(np.where(region_mask))  # Pixel locations

        # Convert raster indices to real-world coordinates
        real_coords = [(xmin + col * pixel_size, ymax + row * pixel_size_y) for row, col in coords]

        # Compute convex hull for the region
        if len(real_coords) > 2:  # Convex hull requires at least 3 points
            convex_polygon = Polygon(real_coords).convex_hull

            # Save polygon to shapefile
            shp.write({"geometry": convex_polygon.__geo_interface__, "properties": {"id": region_id}})

print(f"Convex hulls for {num_regions} regions saved as a shapefile successfully!")

# Close dataset
dataset = None
