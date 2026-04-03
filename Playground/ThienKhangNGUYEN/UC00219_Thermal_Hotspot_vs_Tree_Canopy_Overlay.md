# UC00219 - Thermal Hotspot vs Tree Canopy Overlay

**Authored by:** Thien Khang Nguyen

**Duration:** 90 mins

**Level:** Intermediate

**Pre-requisite Skills:** Python, Data Cleaning, Data Visualisation, Geospatial Analysis, Raster and Vector Data Handling, Spatial Overlay, Feature Interpretation

**Scenario**

As a student interested in urban sustainability and climate resilience in Melbourne, Khang wants to understand which parts of the city may be more vulnerable to urban heat. During hot days, some streets, open spaces, and built-up areas can become much warmer than others, especially where tree canopy is limited. This can reduce outdoor comfort, increase heat exposure, and affect the liveability of urban environments.

Khang wants to have access to a system that can identify Melbourne regions with high surface temperature and low canopy cover by combining thermal imagery with tree canopy spatial data. This would help highlight areas that may benefit from urban greening, improved shade, or future environmental planning. By visualising where thermal hotspots overlap with limited canopy cover, the project can support more informed decisions about climate adaptation and urban design.

**What this use case will teach you**

At the end of this use case you will:
- Learn how to source and combine multiple public geospatial datasets.
- Understand the difference between raster data and vector polygon data in urban analysis.
- Explore how thermal imagery can be used to identify areas of higher surface temperature.
- Analyse tree canopy polygons to understand spatial distribution of urban vegetation.
- Reproject and align datasets into a common coordinate reference system (CRS).
- Apply spatial overlay techniques to compare heat patterns and canopy cover.
- Create a shared analysis unit, such as a grid, to summarise thermal and canopy indicators.
- Identify regions that show the combination of high temperature and low tree canopy cover.
- Visualise spatial patterns on maps to support interpretation and planning discussions.

**Introduction**

Urban heat is an important issue in cities such as Melbourne, where built surfaces like roads, roofs, and pavements can absorb and retain heat. Areas with limited tree canopy often experience stronger heat exposure because there is less shade and less cooling from vegetation. Understanding where high heat and low canopy cover occur together can help reveal locations that may be more vulnerable during hot weather conditions.

This use case focuses on comparing two spatial datasets from the City of Melbourne. The first dataset is a thermal image captured in 2012, which shows differences in surface temperature across the municipality. The second dataset is a tree canopy dataset from 2021, which maps the extent of tree canopy across public and private land.

In this project, the thermal layer will be used to represent surface heat patterns, while the tree canopy layer will be used to represent vegetation coverage. By aligning and overlaying these two datasets, the analysis can identify areas where thermal intensity is high and canopy cover is low.

The datasets used in this project are the "Thermal Image 2012" dataset and the "Tree Canopies 2021 (Urban Forest)" dataset from the City of Melbourne website.
In the notebook implementation, both datasets are accessed through the City of Melbourne API v2.1 endpoints without exposing API keys.

**Important note**

One limitation of this project is that the two datasets come from different years. The thermal image is from 2012, while the tree canopy data is from 2021. Because of this, the results should be interpreted as an exploratory spatial analysis rather than a same-time causal comparison. However, the project can still provide useful insight into broad spatial patterns of heat vulnerability and canopy distribution in Melbourne.
