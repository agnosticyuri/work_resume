# -*- coding: utf-8 -*-
"""
@author: jske
"""
import geopandas as gpd
import re
import numpy as np
#from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from pathlib import Path

class CpPreparation:
    def __init__(self, lot_number, base_dir, radius, min_points_in_cluster, lot_dir, lot_layer_name, cp_points_name):
        # Initialize with a lot number and base directory for file paths.
        self.lot_number = lot_number
        self.base_dir = base_dir
        self.lot_dir = lot_dir
        self.radius = radius
        self.min_points_in_cluster = min_points_in_cluster
        self.polygons = None
        self.points = None
        self.gdf = None
        self.lot_layer_name = lot_layer_name
        self.cp_points_name = cp_points_name

    def check_inputs(self):
        # Validate inputs before processing.
        if not isinstance(self.lot_number, int) or self.lot_number > 73:
            raise ValueError("lot_number must be an integer and lower or equal to index 73.")

        if not isinstance(self.radius, (int, float)) or self.radius <= 0:
            raise ValueError("Radius must be a positive number.")

        if not isinstance(self.min_points_in_cluster, int) or self.min_points_in_cluster <= 0:
            raise ValueError("Minimum points in cluster must be a positive integer.")

    def check_file_exists(self):
        # Check if a file exists using pathlib.
        path = Path(self.base_dir)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {self.base_dir}")
        else:
            print(f"File exists: {self.base_dir}") 

        path2 = Path(self.lot_dir, self.lot_layer_name)
        if not path2.exists():
            raise FileNotFoundError(f"File not found: {self.lot_dir}/{self.lot_layer_name}")
        else:
            print(f"File exists: {self.lot_dir}/{self.lot_layer_name}")   

        path3 = Path(self.lot_dir, self.cp_points_name)
        if not path3.exists():
            raise FileNotFoundError(f"File not found: {self.lot_dir}/{self.cp_points_name}")
        else:
            print(f"File exists: {self.lot_dir}/{self.cp_points_name}")      

    def load_data(self):
        # Load polygon and point shapefiles.
        self.polygons = gpd.read_file(f"{self.lot_dir}/lot_hranice_vol2.shp")
        self.points = gpd.read_file(f"{self.lot_dir}/vyskove_body.shp")

    def filter_polygons(self):
        # Select features based on an attribute.
        return self.polygons[self.polygons["LOT_number"] == self.lot_number]

    def clip_points(self, selected_polygons):
        # Perform spatial intersection to clip points inside selected polygons.
        clipped_points = self.points[self.points.geometry.within(selected_polygons.unary_union)]
        clipped_points.to_file(f"{self.base_dir}/clipped_points_{self.lot_number}.shp")
        self.gdf = gpd.read_file(f"{self.base_dir}/clipped_points_{self.lot_number}.shp")

    def classify_points(self):
        # Classify points into predefined groups based on patterns.
        patterns = ["sm", "nm", "zp", "np", "Asm", "Anm", "Azp", "Anp"]
        self.gdf["group"] = self.gdf["CISLO_BODU"].apply(lambda x: next((p for p in patterns if p in x), "No_pattern"))

    def apply_dbscan(self):
        # Apply DBSCAN clustering using user-input radius and min_points_cluster.
        def cluster_group(group):
            coords = np.array(list(zip(group.geometry.x, group.geometry.y)))
            db = DBSCAN(eps=self.radius, min_samples=self.min_points_in_cluster).fit(coords)
            group["cluster"] = db.labels_
            return group

        group_column = self.gdf["group"]
        gdf_without_group = self.gdf.drop(columns=["group"])

        clustered_gdf = gdf_without_group.groupby(group_column, group_keys=False).apply(cluster_group).reset_index(drop=True)
        clustered_gdf["group"] = group_column  
        self.gdf = clustered_gdf

    """
    # Alternative DBSCAN implementation (not used)
    def apply_dbscan(self):
        def cluster_group(group):
            coords = np.array(list(zip(group.geometry.x, group.geometry.y)))
            db = DBSCAN(eps=self.radius, min_samples=self.min_points_in_cluster).fit(coords)
            db.labels_ = db.labels_ + 1
            group["cluster"] = db.labels_
            return group

        try:
            self.gdf = self.gdf.groupby("group", group_keys=False, include_groups=False).apply(cluster_group)
        except TypeError:
            self.gdf = self.gdf.groupby("group", group_keys=False).apply(lambda x: cluster_group(x.reset_index(drop=True)))
    """    

    def rename_points(self):
        # Rename point attributes using defined transformation rules.
        self.gdf["CISLO_BODU"] = self.gdf.apply(lambda row: self.replace_second_xx(row["CISLO_BODU"], row["cluster"]), axis=1)
        self.gdf["CISLO_BODU"] = self.gdf["CISLO_BODU"].apply(self.format_second_number)
        self.gdf["CISLO_BODU"] = [self.rename_first_digits(value) for value in self.gdf["CISLO_BODU"]]

    def save_results(self):
        # Save updated shapefile and formatted text file.
        self.gdf.to_file(f"{self.base_dir}/clipped_renamed_checked{self.lot_number}.shp", driver="ESRI Shapefile")
        self.save_text_file()

# Provide directory for saving outputs
#base_dir = "D:/rename_vertical_accuracy"
base_dir = "D:/cpcp"
# Path to lot borders and cp point layer shapefile
lot_dir = "D:/rename_vertical_accuracy"
lot_layer_name="lot_hranice_vol2.shp"# lot borders shapefile name
cp_points_name="vyskove_body.shp"# cp points shapefile name
# if lot number is 1 digit write it as two digits form: 01, 02, 03 etc.
lot_number = 16
# Optional param for DBSCAN should be fixed change only in case of a problem, minimal points to form a single cluster is always 4
radius=30
min_points_in_cluster=4
#processing
prep = CpPreparation(lot_number, base_dir, radius, min_points_in_cluster, lot_dir,lot_layer_name,cp_points_name)
prep.check_inputs()
prep.check_file_exists()
prep.load_data()
selected_polygons = prep.filter_polygons()
prep.clip_points(selected_polygons)
prep.classify_points()
prep.apply_dbscan()
prep.rename_points()
prep.save_results()