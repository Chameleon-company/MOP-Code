import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from fuzzywuzzy import process, fuzz
import re

# Load data from specific files
health_behavior = pd.read_csv("C:/Users/akind/OneDrive/Desktop/782 work/nancy work/HB1.csv")
crime_data = pd.read_excel("C:/Users/akind/OneDrive/Desktop/782 work/nancy work/Crime_data.xlsx", sheet_name='Table 01')

# Clean column names: strip spaces and convert to lowercase
health_behavior.columns = health_behavior.columns.str.strip().str.lower()
crime_data.columns = crime_data.columns.str.strip().str.lower()

# Standardize location names (remove leading/trailing spaces and convert to lowercase)
health_behavior['location'] = health_behavior['location'].str.strip().str.lower()
crime_data['location'] = crime_data['location'].str.strip().str.lower()

# Define a function to get the best match for a location
def get_best_match(location, choices):
    match, score = process.extractOne(location, choices, scorer=fuzz.partial_ratio)
    return match

# Get a list of unique locations from crime data
crime_locations = crime_data['location'].unique()

# Match locations in health_behavior to crime_data using fuzzy matching
health_behavior['matched_location'] = health_behavior['location'].apply(lambda loc: get_best_match(loc, crime_locations))

# Merge datasets on the best match location
data = pd.merge(health_behavior, crime_data, left_on='matched_location', right_on='location', how='left', suffixes=('_hb', '_crime'))

# Handle missing values in crime data
data['crime_count'].fillna(0, inplace=True)

# Calculate Perceived Safety Score
data['perceived_safety_score'] = (data['crime_count'] / data['rate per 100,000 pop']) * 100

# Define feature columns and target
X = data[['age', 'gender', 'matched_location', 'likelihood_percent']]
y = data['behavior']  # Assuming 'behavior' is the target

# Preprocessing for numeric and categorical features
numeric_features = ['likelihood_percent']
categorical_features = ['age', 'gender', 'matched_location']

# Create preprocessing pipelines
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and configure the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Define an expanded parameter grid for RandomizedSearchCV
param_dist = {
    'classifier__n_estimators': [50, 100, 200, 300, 400, 500],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_depth': [None, 10, 20, 30, 40, 50],
    'classifier__min_samples_split': [2, 5, 10, 15],
    'classifier__min_samples_leaf': [1, 2, 4, 6],
    'classifier__bootstrap': [True, False]
}

# Set up the RandomizedSearchCV with more iterations
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,  # Increase the number of fits
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
random_search.fit(X_train, y_train)

# Get the best model
best_rf_model = random_search.best_estimator_

# Evaluate the model
y_pred = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Calculate model accuracy
accuracy = best_rf_model.score(X_test, y_test)
accuracy_percentage = accuracy * 100
print(f"Model Accuracy: {accuracy_percentage:.2f}%")

# Function to parse user query and make predictions
def parse_and_predict(age, gender, location, likelihood_percent):
    # Create a DataFrame from the user input
    user_input = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'matched_location': [get_best_match(location, crime_locations)],
        'likelihood_percent': [likelihood_percent]
    })
    
    # Make prediction
    prediction = best_rf_model.predict(user_input)
    return prediction[0]

# Function to calculate Perceived Safety Score
def calculate_perceived_safety_score(location):
    # Standardize the location name
    location = location.strip().lower()
    matched_location = get_best_match(location, crime_locations)
    
    # Check if the matched location is in the crime data
    if matched_location in crime_data['location'].values:
        predicted_location_data = crime_data[crime_data['location'] == matched_location]
        crime_count = predicted_location_data['crime_count'].values[0]
        rate_per_100000_pop = predicted_location_data['rate per 100,000 pop'].values[0]
        
        # Ensure rate_per_100000_pop is not zero to avoid division by zero
        if rate_per_100000_pop != 0:
            perceived_safety_score = (crime_count / rate_per_100000_pop) * 100
        else:
            perceived_safety_score = 0
        
        return perceived_safety_score
    else:
        return "Location not found in crime data."

# Function to get safety feedback
def get_safety_feedback(score):
    # Define a threshold for safety
    safety_threshold = 50  # Adjust this threshold based on your criteria
    
    if isinstance(score, str):  # Check if score is an error message
        return score
    elif score >= safety_threshold:
        return "The location is generally safe at night."
    else:
        return "The location might be unsafe at night."

# Function to handle queries
def handle_query(age, gender, location, likelihood_percent):
    predicted_behavior = parse_and_predict(age, gender, location, likelihood_percent)
    print(f'Predicted Behavior: {predicted_behavior}')
    
    # Calculate perceived safety score
    perceived_safety_score = calculate_perceived_safety_score(location)
    print(f'Perceived Safety Score for {location}: {perceived_safety_score}')
    
    # Get safety feedback
    safety_feedback = get_safety_feedback(perceived_safety_score)
    print(f'Safety Feedback: {safety_feedback}')

# Function to parse the question
def parse_question(question):
    # Adjust regex patterns to capture more flexible formats
    age_pattern = r"age\s*=\s*'([^']+)'"
    gender_pattern = r"gender\s*=\s*'([^']+)'"
    location_pattern = r"location\s*=\s*'([^']+)'"
    likelihood_pattern = r"likelihood_percent\s*=\s*([\d.]+)"

    age_match = re.search(age_pattern, question, re.IGNORECASE)
    gender_match = re.search(gender_pattern, question, re.IGNORECASE)
    location_match = re.search(location_pattern, question, re.IGNORECASE)
    likelihood_match = re.search(likelihood_pattern, question, re.IGNORECASE)

    if age_match:
        age = age_match.group(1).strip()
    else:
        age = None

    if gender_match:
        gender = gender_match.group(1).strip()
    else:
        gender = None

    if location_match:
        location = location_match.group(1).strip()
    else:
        location = None
    
    if likelihood_match:
        try:
            likelihood_percent = float(likelihood_match.group(1).strip())
        except ValueError:
            likelihood_percent = None
    else:
        likelihood_percent = None
    
    # Check if all necessary information is found
    if age and gender and location and likelihood_percent is not None:
        handle_query(age, gender, location, likelihood_percent)
    else:
        print("Could not parse the question. Ensure it is formatted correctly.")

# Example usage with a question
question = "What is the safety prediction for age = '24-35', gender = 'female', living in location = 'carlton' with likelihood_percent = 18.2?"
parse_question(question)


# In[21]:





# In[ ]:




