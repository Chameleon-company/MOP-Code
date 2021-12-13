<img src="/images/mop-black.png" alt="drawing" width="100"/>

# Melbourne Open Data Playground
## SETTING UP A CONDA ENVIRONMENT IN ANACONDA

Last Updated: 9-Dec-2021

### Installing the environment
#### Pre-requisites
- Anaconda installed on Windows Environment
- [Copy this conda environment configuration file](environment.yml) to a local directory

#### Installation steps
To setup a new environment with Conda follow these steps:

1) Launch Anaconda

2) Goto to the "base" environment

3) Launch Command Prompt

4) Run the following commands:

&emsp;&emsp;&gt;cd {directory containing the environment.yml file}

&emsp;&emsp;&gt;conda env create -f environment.yml

5) In Anaconda install JupyterNotebook


### Removing thee environment
To remove the environment repeat steps 1 through 3 above then run the following command

&emsp;&emsp;&gt;conda env remove -n MelbourneCityOpenData

### Environment Packages
The following packages make up the MelbourneCityOpenData data science environment.<br>
Many of these packages are also used in the Melbourne Open Data Playground website.

#### CORE
**requests**<br>
A common library for managing HTTP requests.<br>

#### DATA ENGINEERING
**numpy**<br>
Numpy is a library designed for efficient array computation.<br>
**pandas**<br>
Data structures for data analysis, time series, and statistics.<br>

#### DATA SCIENCE/ML
**scikit-learn**<br>
Sci-kit Learn is a popular Machine/Statitical learning package.<br>

#### VISUALISATION AND IMAGE PROCESSING
**matplotlib**<br>
MatplotLib is a plotting Library for static, animated, and interactive visualizations.<br>
**seaborn**<br>
Seaborn extends MatPlotlib with more sophisticated plot types.<br>
**plotly**<br>
Plotly allows the creation of interactive graphs, plots and maps.<br>

#### OPEN DATA
**sodapy**<br>
Sodapy is a Python interface to the SOCRATA API for accessing Open Data.<br>

#### GEOSPATIAL
**pyproj**<br>
An interface to PROJ (cartographic projections and coordinate transformations library).<br>
**shapely**<br>
Shapely supports manipulation and analysis of geometric objects in the Cartesian plane.<br>
**geopandas**<br>
Geopandas extends the datatypes used by Pandas to allow spatial operations on geometric types.<br>
Geometric operations are performed by shapely.<br>
Geopandas further depends on fiona for file access and matplotlib for plotting.<br>
**geopy**<br>
A Python client for several popular geocoding web services.<br>
Geopy includes geocoder classes for the OpenStreetMap Nominatim, Google Geocoding API (V3), and many other geocoding services.
**folium**<br>
Folium makes it easy to visualize data that’s been manipulated in Python on an interactive leaflet map. It enables both the binding of data to a map for choropleth visualizations as well as passing rich vector/raster/HTML visualizations as markers on the map.<br>
The library has a number of built-in tilesets from OpenStreetMap, Mapbox, and Stamen, and supports custom tilesets with Mapbox or Cloudmade API keys. folium supports both Image, Video, GeoJSON and TopoJSON overlays.<br>