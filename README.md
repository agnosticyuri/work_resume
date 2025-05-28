# work_resume

Summary of point cloud processing and preprocessing scripts for PointNET++ training and BIM (Building information model) creation via rooftop extraction from each LOT region.
PointCNN is just experimental not approved version.
Preprocessing steps include calculation of PCA, features in 3D space like sphericity, linearity, planarity etc.
Clustering methods - mostly DBSCAN, grid based methos like moving  average window for digital terrain model (DTM) processing.
In the future I am planning to use pyspark or dask for establishing local workers and process massive point clouds. Utilizing parallel computing principles for time efficiency.
Possible merging feature and object extraction by using state owned orthophotography data with point clouds. Extracting paved roads, electric powerlines etc. Extracted objects and features could be provided for the public usage as a map layer or other specific data form for local administration and land use management and navigation.

Data for the training are provided by GKU.
