Real-Time Route Optimisation using Traffic & Environmental Data

Overview:
This project focuses on developing a data-driven approach to real-time route optimisation by analysing traffic activity and environmental conditions in Melbourne. The goal is to predict traffic congestion patterns and use these insights to support smarter routing decisions.

Objective:
To build a system that:
Predicts traffic volume (congestion levels) across different locations and times
Incorporates weather and environmental factors
Enables comparison of routes to select the least congested path

Datasets Used:
1. Transport Activity Count Dataset
Provides vehicle counts across multiple locations at 5-minute intervals
Includes different transport types (cars, buses, trucks, etc.)
2. ICT Microclimate Sensor Dataset
Provides environmental data such as:
Temperature
Wind speed
Humidity
Air quality 
Noise levels

Data Processing:
The datasets were preprocessed to ensure consistency and usability:
Filtered relevant vehicle types for congestion analysis
Converted timestamps and aggregated data into hourly intervals
Engineered time-based features (hour, day of week, weekend)
Cleaned and aggregated environmental data across sensors
Datasets were merged on hourly timestamps after aligning time zones and aggregating sensor readings across locations

Data Integration:
Traffic and environmental datasets were merged using timestamp-based joins, creating a unified dataset that captures:
Traffic conditions
Environmental influences
Temporal patterns

This enables analysis of how different factors impact congestion.

Use Case

The processed dataset can be used to:

Train models to predict traffic volume
Identify congestion patterns over time
Support route optimisation systems by:
Comparing predicted congestion across routes
Selecting the most efficient path