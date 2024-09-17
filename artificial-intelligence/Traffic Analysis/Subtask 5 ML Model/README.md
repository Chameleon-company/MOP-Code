# Predicting Vehicle Composition on Road Segments Using Multinomial Logistic Regression

## Project Overview
The goal of this project is to predict the vehicle composition on various road segments, identifying whether a segment is likely to experience "Heavy Traffic," "Light Traffic," or "Mixed Traffic." Using Multinomial Logistic Regression, the model learns the relationships between various road segment features and predicts the likelihood of each traffic composition class.

## Model Details
The model chosen for this task is **Multinomial Logistic Regression** because it is specifically designed for multi-class classification problems. It helps in understanding how features like speed limit, vehicle counts, time of day, and road location influence the likelihood of each traffic class.

## Input and Target Variables
- **Target Variable (y):** Vehicle composition - "Heavy Traffic," "Light Traffic," or "Mixed Traffic"
- **Input Variables (X):** Features used for prediction include:
  - Speed Limit
  - Average Speed
  - Latitude and Longitude
  - Road Characteristics (name, suburb, location)
  - Direction
  - Vehicle Ratios (ratios of light and heavy vehicles)

## Data Preprocessing
### Defining Light and Heavy Vehicles
- Calculated proportions of light and heavy vehicles for each road segment.
- Created a target variable (Vehicle Composition) based on these ratios:
  - "Heavy Traffic": Heavy vehicles > 50%
  - "Light Traffic": Light vehicles > 70%
  - "Mixed Traffic": Otherwise

### Data Splitting and Standardization
- The dataset was split into training (80%) and testing (20%) sets.
- Standardization was applied to input features to have a mean of 0 and a standard deviation of 1.

## Model Training
The model was trained using the **Multinomial Logistic Regression** algorithm:
- `multi_class=‘multinomial’`
- `max_iter=1000`

## Results and Evaluation
### Coefficients
- Coefficients indicate the influence of each feature on predicting each traffic class.
- Positive coefficients imply an increase in the probability of the respective class as the feature value increases.

### Classification Report
- The classification report provides metrics like precision, recall, and F1-score for each traffic category:
  - **Heavy Traffic**: Low precision but high recall
  - **Light Traffic**: High precision and recall
  - **Mixed Traffic**: Moderate performance

### Confusion Matrix
- Visualizes the number of correct and incorrect predictions.
- Indicates areas where the model is confused in its predictions.

## Images and Visualizations
Below are some visualizations used in this project:

### Confusion Matrix
![Confusion Matrix](artificial-intelligence/Traffic Analysis/Subtask 5 ML model/confusion_matrix.png)

### Example Model Output
![Model Output](artificial-intelligence/Traffic Analysis/Subtask 5 ML model/model_output.png)

> **Note:** Replace `path_to_image` with the actual path to the images in your repository.

## Cross-Validation
- Applied 5-fold cross-validation.
- Average cross-validation score: 73.63%.

## Real-World Application
The model can be used for real-time traffic management, predicting the likelihood of heavy traffic on major highways during rush hours and helping in infrastructure planning.

## Future Work
- Handle data imbalance for better prediction of heavy traffic.
- Incorporate time-based predictions for each road segment.

## How to Run
1. Clone this repository.
2. Install required packages (`numpy`, `pandas`, `sklearn`, etc.).
3. Run the provided Python script/notebook to preprocess data, train the model, and evaluate results.

