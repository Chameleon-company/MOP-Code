\# README.md



```markdown



\# Melbourne Traffic Accident Risk Prediction



\## Project Overview

This project aims to predict traffic accident risk at urban intersections in Melbourne using machine learning techniques.  

By analysing historical crash data together with environmental and mobility indicators, the project identifies patterns associated with higher accident probability.



The system integrates multiple datasets including traffic crash records, weather conditions, pedestrian activity, and bicycle traffic.



The final output includes predictive models, data visualisations, and a conceptual dashboard designed to support urban traffic safety analysis.



\---



\## Project Objectives



The primary objectives of this project are:



\- Identify high-risk intersections using historical crash data

\- Analyse accident patterns across time and environmental conditions

\- Examine the relationship between weather conditions and accident frequency

\- Incorporate pedestrian and bicycle activity data into accident risk modelling

\- Develop machine learning models to predict accident risk

\- Create visualisations and a dashboard to communicate insights



\---



\## Team Members



Suba Thinakaran

Khalid Ameen

Burhanuddin Ujjainwala



\---



\## Datasets Used



\### 1. Victorian Road Crash Data

Source: Victoria Government Open Data



Contains detailed records of traffic accidents including:

\- crash date and time

\- location coordinates

\- crash severity

\- road characteristics

\- speed zones



\---



\### 2. Melbourne Weather Data

Source: Bureau of Meteorology / Melbourne Government



Weather variables include:

\- rainfall

\- temperature

\- wind speed

\- weather conditions



Weather conditions are integrated to analyse environmental effects on accident risk.



\---



\### 3. Melbourne Pedestrian Counts (MOP)

Source: City of Melbourne Open Data Portal



Dataset contains:

\- hourly pedestrian counts

\- sensor locations

\- timestamped observations



\---



\### 4. Melbourne Bicycle Counts (MOP)

Source: City of Melbourne Open Data Portal



Dataset includes:

\- cyclist activity measurements

\- timestamps

\- sensor locations



\---



\## Project Workflow



The project follows a typical data science workflow:



1\. Data Collection  

2\. Data Cleaning  

3\. Exploratory Data Analysis  

4\. Feature Engineering  

5\. Machine Learning Modelling  

6\. Model Evaluation  

7\. Visualisation and Dashboard Design



\---



\## Repository Structure



```



Traffic\_Accident\_Prediction

│

├── 01\_data\_collection\_mop.ipynb

├── 02\_crash\_data\_import.ipynb

├── 03\_weather\_data\_import.ipynb

├── 04\_mop\_data\_cleaning.ipynb

├── 05\_crash\_data\_cleaning.ipynb

├── 06\_weather\_data\_cleaning.ipynb

├── 07\_mobility\_analysis.ipynb

├── 08\_crash\_analysis.ipynb

├── 09\_weather\_crash\_analysis.ipynb

├── 10\_mobility\_features.ipynb

├── 11\_crash\_features.ipynb

├── 12\_weather\_features.ipynb

├── 13\_dataset\_merge.ipynb

├── 16\_baseline\_models.ipynb

├── 17\_advanced\_models.ipynb

└── README.md



```



Each notebook focuses on a specific stage of the analysis to ensure collaboration efficiency and avoid editing conflicts.



\---



\## Machine Learning Models



The following models will be explored:



\- Logistic Regression

\- Decision Tree

\- Random Forest

\- Gradient Boosting / XGBoost



Model performance will be evaluated using:



\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- Confusion Matrix



\---



\## Expected Outcomes



The project aims to produce:



\- insights into accident patterns in Melbourne

\- identification of high-risk intersections

\- a predictive model for accident risk

\- visual dashboards for communicating findings



Example prediction output:



```



Location: Swanston Street

Time: 17:00

Weather: Rain

Pedestrian Activity: High



Predicted Accident Risk: High



```



\---



\## Dashboard Concept



The final dashboard will present:



\- accident hotspots across Melbourne

\- accident trends by time and weather

\- feature importance from the machine learning model

\- an interactive accident risk prediction tool



\---



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib / Seaborn

\- Jupyter Notebook

\- Figma (for dashboard design)



\---



\## Project Timeline



The project follows an 8-week development plan including:



\- data acquisition

\- data processing

\- model development

\- visualisation

\- final reporting



\---



\## Repository Usage



Team members should follow this workflow:



1\. Pull latest changes from GitHub

2\. Work on assigned notebook files

3\. Commit changes locally

4\. Push updates to the project branch



This ensures smooth collaboration and avoids merge conflicts.



\---



\## Future Improvements



Possible extensions include:



\- incorporating traffic volume datasets

\- using geospatial modelling techniques

\- developing a real-time accident risk prediction system



\---



\## License



This project is developed for academic purposes as part of a university data science project in collaboration with the City of Melbourne.

```



\---





