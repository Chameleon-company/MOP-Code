=====================================================
SETTING UP "MelbourneCityOpenData" CONDA ENVIRONMENT
FOR USE WITH Anaconda and Jupyter Notebook
=====================================================
Prepared by Steven Tuften <stuften@deakin.edu.au>
Last Updated: 10-Nov-2021
-----------------------------------------------------

To setup a new environment with conda..

1) Launch Anaconda

2) Goto to the "base" environment

3) Launch Command Prompt

4) Run the following commands:

cd {directory containing the environment.yml file}

conda env create -f environment.yml

5) In Anaconda install JupyterNotebook


=====================================================
To remove the environment repeat steps 1 through 3 above then run the following command

conda env remove -n MelbourneCityOpenData

=====================================================
The following packages make up the MelbourneCityOpenData environment.

# CORE
requests	# HTTP Library

# DATA ENGINEERING
numpy		# Array computation
pandas		# Data structures for data analysis, time series, and statistics

# DATA SCIENCE/ML
scikit-learn	# Machine/Statitical learning

# VISUALISATION AND IMAGE PROCESSING
matplotlib	# Plotting Library for static, animated, and interactive visualizations
seaborn		# Extends MatPlotlib with more sophisticated plot types
plotly		# Create interactive graphs in Jupyter notebook

# OPEN DATA
sodapy		# python interface to SOCRATA API

# GEOSPATIAL
pyproj		# interface to PROJ (cartographic projections and coordinate transformations library)
shapely		# Manipulation and analysis of geometric objects in the Cartesian plane.
geopandas	# Extends the datatypes used by Pandas to allow spatial operations on geometric types.
		# Geometric operations are performed by shapely.
		# Geopandas further depends on fiona for file access and matplotlib for plotting.
geopy		# Python client for several popular geocoding web services.
		# geopy includes geocoder classes for the OpenStreetMap Nominatim, Google Geocoding API (V3), and many other geocoding services.
folium		# folium makes it easy to visualize data that