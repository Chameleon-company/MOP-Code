#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns

# Load data
file_path = r"C:\Users\akind\OneDrive\Desktop\782 work\HB1.csv"
data = pd.read_csv(file_path)

# Map gender to numeric
data['gender'] = data['gender'].map({'male': 0, 'female': 1})

# Convert location to numeric codes
data['location_code'] = data['location'].astype('category').cat.codes

# Encode age ranges to numeric categories
data['age_code'] = data['age'].astype('category').cat.codes

# Ensure the description column is included in the dataset
description_dict = {
    'Mental Health': 'Excellent',
    'Physical Health': 'Excellent',
    'Smoking': 'Smoking daily',
    'Vaping': 'Vaping daily.'
}
data['description'] = data['behavior'].map(description_dict)

# Define features and target variable
X = data[['age_code', 'gender', 'location_code', 'likelihood_percent']]
y = data['behavior']

# Handle missing values by imputing with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Define Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Perform Randomized Search to find the best hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, 
                                      n_iter=20, cv=5, verbose=1, random_state=42, n_jobs=-1)

# Fit the model
rf_random_search.fit(X_train, y_train)

# Best model from random search
best_rf_model = rf_random_search.best_estimator_

# Define function to get description based on probability
def get_description_based_on_probability(probability):
    if probability < 0.2:
        return 'Bad'
    elif 0.2 <= probability < 0.4:
        return 'Fair'
    elif 0.4 <= probability < 0.7:
        return 'Good'
    else:
        return 'Excellent'

# Function to predict likelihood and generate description
def predict_likelihood(age, gender, location_likelihood_map):
    # Convert inputs to numeric values
    gender = 0 if gender == 'male' else 1
    
    # Map location and age to numeric codes
    location_categories = data['location'].astype('category').cat.categories.tolist()
    age_categories = data['age'].astype('category').cat.categories.tolist()
    
    if age not in age_categories:
        raise ValueError(f"Age '{age}' not found in dataset. Available age groups are: {age_categories}")
    
    age_code = age_categories.index(age)
    
    # Get the behavior categories
    behavior_categories = data['behavior'].astype('category').cat.categories.tolist()
    
    results = []
    for location, likelihood_percent in location_likelihood_map.items():
        if location not in location_categories:
            raise ValueError(f"Location '{location}' not found in dataset. Available locations are: {location_categories}")
        
        location_code = location_categories.index(location)
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[age_code, gender, location_code, likelihood_percent]], 
                                  columns=['age_code', 'gender', 'location_code', 'likelihood_percent'])
        
        # Impute and scale features
        scaled_features = scaler.transform(imputer.transform(input_data))
        
        # Get the probabilities for all classes
        probabilities = best_rf_model.predict_proba(scaled_features)[0]
        
        # Find the behavior with the highest probability
        max_prob_index = probabilities.argmax()
        behavior = behavior_categories[max_prob_index]
        likelihood = probabilities[max_prob_index]
        
        # Get description
        description = description_dict.get(behavior, 'No description available')

        # Get prediction quality based on likelihood
        prediction_quality = get_description_based_on_probability(likelihood)
        
        results.append({
            'Age': age,
            'Location': location,
            'Likelihood %': likelihood_percent,
            'Predicted Behavior': behavior,
            'Probability': likelihood,
            'Condition': description,
            'Prediction Quality': prediction_quality
        })
    
    return pd.DataFrame(results)

# Define the specific parameters for prediction
specific_age = '55-64'
specific_location_likelihood_map = {
    'Docklands': 47.5,
    'Carlton': 45.8,
    'Melbourne CBD': 10.5
}

# Make predictions for the specific parameters
results_df = predict_likelihood(specific_age, 'female', specific_location_likelihood_map)

# Display results in a table
print("Prediction Results:")
print(results_df.to_string(index=False))

# Plotting
plt.figure(figsize=(14, 8))

# Use 'Location' as x-axis, 'Probability' as y-axis, and color by 'Prediction Quality'
sns.barplot(data=results_df, x='Location', y='Probability', hue='Predicted Behavior', palette='viridis')

plt.xlabel('Location')
plt.ylabel('Probability')
plt.title(f'Predicted Behavior Probability for Age: {specific_age}')
plt.legend(title='Predicted Behavior')
plt.show()



# In[28]:


# Model Evaluation
# Predict on the test set
y_pred = best_rf_model.predict(X_test)
y_proba = best_rf_model.predict_proba(X_test)  # Probabilities for all classes

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

# Print performance metrics
print("\nModel Evaluation:")
metrics = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    "Value": [accuracy, precision, recall, f1, roc_auc]
}
print(tabulate(metrics, headers="keys", tablefmt="grid"))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=best_rf_model.classes_, columns=best_rf_model.classes_)
print("\nConfusion Matrix:")
print(tabulate(conf_matrix_df, headers="keys", tablefmt="grid"))

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=best_rf_model.classes_)
print("\nClassification Report:")
print(class_report)


# In[30]:


# Identify location with the lowest probability for smoking
location_with_min_smoking_prob = results_df.loc[results_df['Probability of Smoking'].idxmin()]

print("\nLocation with Decreased Smoking Behavior:")
print(location_with_min_smoking_prob.to_string(index=False))

# Plotting
plt.figure(figsize=(14, 8))
sns.barplot(data=results_df, x='Location', y='Probability of Smoking', palette='viridis')
plt.xlabel('Location')
plt.ylabel('Probability of Smoking')
plt.title(f'Probability of Smoking for Age: {specific_age}')
plt.show()


# In[31]:


# Identify location with the highest probability for smoking
location_with_max_smoking_prob = results_df.loc[results_df['Probability of Smoking'].idxmax()]

print("\nLocation with Increased Smoking Behavior:")
print(location_with_max_smoking_prob.to_string(index=False))

# Plotting
plt.figure(figsize=(14, 8))
sns.barplot(data=results_df, x='Location', y='Probability of Smoking', palette='viridis')
plt.xlabel('Location')
plt.ylabel('Probability of Smoking')
plt.title(f'Probability of Smoking for Age: {specific_age}')
plt.show()


# In[ ]:




