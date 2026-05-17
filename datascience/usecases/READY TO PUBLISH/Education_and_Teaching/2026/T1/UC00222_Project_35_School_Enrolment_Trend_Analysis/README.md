# UC00222 Project 35 - School Enrolment Trend Analysis

## Overview

This use case explores Victorian school enrolment trends from 2022 to 2025.

The analysis combines annual school enrolment CSV files with Victorian school location and profile data to investigate how enrolment changes over time across education sectors, school types, year levels, regions, and local government areas. The project includes data loading, cleaning, feature engineering, exploratory data analysis, visualisation, heatmaps, school-level change analysis, predictive modelling, model evaluation, and a cautious 2026 enrolment forecast.

## Use Case Scenario

As education planning analysts, we want to combine Victorian school enrolment data from 2022 to 2025 with school location information so that we can identify where enrolment is growing, where it is declining, and how those patterns differ by sector, school type, year level, region, and local government area.

## Datasets Used

### 1. Victorian School Enrolment Dataset

The school enrolment dataset is sourced from the Chameleon-company MOP-Code GitHub repository.

The project uses four annual CSV files covering 2022, 2023, 2024, and 2025. These files are loaded directly from the raw GitHub URLs so the notebook can be run in Google Colab without requiring manual file uploads.

The dependency files are stored in the shared repository dependency location:

```text
datascience/usecases/DEPENDENCIES/UC00206_Project 35_Source Data/
```

GitHub source folder:

```text
https://github.com/Chameleon-company/MOP-Code/tree/master/datascience/usecases/DEPENDENCIES/UC00206_Project%2035_Source%20Data
```

Files used:

```text
dv335-allschoolsFTEenrolmentsFeb2022.csv
dv355-VIC All Schools Enrolments 2023.csv
dv377_DataVic-AllSchoolsEnrolments-2024.csv
dv403-AllSchoolsEnrolments-2025.csv
```

### 2. Victorian School Location/Profile Dataset

The school location and profile dataset is sourced from the Victorian Government open data API.

This dataset provides school profile fields such as school name, education sector, school type, region, local government area, address information, and coordinates. These fields are merged with the enrolment data using a stronger school identifier created from `Entity_Type` and `School_No`.


```text
https://discover.data.vic.gov.au/dataset/school-locations-2024
```


## Project Workflow

The notebook follows this workflow:

1. Load annual enrolment CSV files from GitHub.
2. Load school location/profile data from the Victorian Government API.
3. Clean column names and remove encoding artefacts.
4. Compare annual file structures and keep shared columns.
5. Check duplicate school numbers and school status values.
6. Create `Unique_School_ID` using `Entity_Type` and `School_No`.
7. Filter to schools that are open and present in all four years.
8. Merge enrolment data with school location/profile data.
9. Validate the final dataset.
10. Create trend summaries, visualisations, heatmaps, and school-level change tables.
11. Train and evaluate a baseline predictive model.
12. Produce a cautious 2026 enrolment forecast.
13. Summarise key findings, limitations, and references.

## Analysis and Visualisations

The notebook includes:

- workflow diagram
- overall enrolment trend line charts
- average enrolment per school chart
- education sector trend chart
- school type stacked bar chart
- year-level enrolment heatmap
- regional enrolment heatmap
- LGA growth and decline chart
- top school increases and decreases
- distribution of school-level enrolment change
- spatial scatter plot using school coordinates
- correlation heatmap
- predicted vs actual model validation chart
- model prediction error distribution chart
- actual 2025 vs forecast 2026 sector chart

## Predictive Model

The project includes a baseline supervised learning model using Ridge regression.

The model predicts latest-year enrolment using:

- year index
- previous-year enrolment
- previous-year enrolment change
- education sector
- school type
- Department of Education administrative region

The model is trained on earlier school-year records and validated against 2025 actual enrolment values. It is also compared with a simple previous-year baseline. Because only four annual snapshots are available, the model is treated as an indicative baseline rather than a high-certainty formal projection.

The notebook also produces a cautious 2026 forecast at school level and sector level.

## Key Outputs

The final notebook produces:

- final merged dataset with 8,984 rows
- school trend table covering 2,246 consistent open schools
- yearly enrolment summary
- sector and regional trend summaries
- school-level increase and decrease tables
- model evaluation metrics
- 2026 school-level forecast
- 2026 sector-level forecast
- final findings table

## References

- Chameleon-company MOP-Code GitHub repository, UC00206 Project 35 Source Data.
- Victorian Government open data API, school location/profile resource `https://discover.data.vic.gov.au/dataset/school-locations-2024`
