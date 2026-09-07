#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install gradio')


# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import gradio as gr
import difflib

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

# Initialize state variables
state = {"age": None, "gender": None, "location_likelihood_map": {}, "name": None, "action": None}

def correct_location_name(location, location_categories):
    """Corrects the location name based on the closest match from the dataset."""
    close_matches = difflib.get_close_matches(location, location_categories, n=1, cutoff=0.6)
    if close_matches:
        return close_matches[0]
    else:
        raise ValueError(f"Location '{location}' not found in dataset. Available locations are: {location_categories}")

def get_age_code(age_input):
    """Convert age input to age code based on dataset categories."""
    age_categories = data['age'].astype('category').cat.categories.tolist()
    
    # Normalize age categories by removing trailing spaces
    normalized_age_categories = [age.strip() for age in age_categories]
    
    # Convert age_input to integer if it's a single age
    if age_input.isdigit():
        age_input = int(age_input)
        for age_range in normalized_age_categories:
            start_age, end_age = age_range.split('-')
            if start_age.isdigit() and end_age.isdigit():
                if age_input in range(int(start_age), int(end_age) + 1):
                    return normalized_age_categories.index(age_range)
        raise ValueError(f"Age '{age_input}' not found in dataset. Available age groups are: {normalized_age_categories}")
    else:
        # Check for exact matches with age ranges
        if age_input in normalized_age_categories:
            return normalized_age_categories.index(age_input)
        else:
            raise ValueError(f"Age '{age_input}' not found in dataset. Available age groups are: {normalized_age_categories}")

def predict_likelihood(age_input, gender, location_likelihood_map):
    """Predict likelihood and generate description for multiple locations."""
    # Convert inputs to numeric values
    gender = 0 if gender == 'male' else 1
    
    # Map location and age to numeric codes
    location_categories = data['location'].astype('category').cat.categories.tolist()
    
    age_code = get_age_code(age_input)
    
    # Get the behavior categories
    behavior_categories = data['behavior'].astype('category').cat.categories.tolist()
    
    results = []
    for location, likelihood_percent in location_likelihood_map.items():
        # Correct the location name
        corrected_location = correct_location_name(location, location_categories)
        
        location_code = location_categories.index(corrected_location)
        
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

        # Determine feedback
        feedback = 'Good' if likelihood > 0.6 else ('Fair' if likelihood > 0.4 else 'Bad')

        # Format output based on behavior
        if behavior == 'Smoking':
            behavior_description = 'Smoke daily'
        elif behavior == 'Vaping':
            behavior_description = 'Vaping daily.'
        else:
            behavior_description = description
        
        results.append({
            'Age': age_input,
            'Location': corrected_location,
            'Predicted Behavior': behavior,
            'Probability': likelihood,
            'Condition': behavior_description,
            'Gender': 'Male' if gender == 0 else 'Female',
            'Feedback': feedback
        })
    
    return pd.DataFrame(results)

def iterate_input(query):
    """Process user input and guide through the chatbot conversation."""
    if state["name"] is None:
        state["name"] = query.strip()
        return "Hello {name}! What would you like to do today? Type 'predict' to start the behavior prediction or 'reset' to start over.".format(name=state["name"])
    
    if state["action"] is None:
        if query.strip().lower() == "predict":
            state["action"] = "predict"
            return "Please provide your age:"
        elif query.strip().lower() == "reset":
            reset_state()
            return start_prompt()
        else:
            return "Invalid choice. Please type 'predict' to start the behavior prediction or 'reset' to start over."
    
    if state["age"] is None:
        age_input = query.strip()
        try:
            age_code = get_age_code(age_input)
            state["age"] = age_input
            return "Please provide the gender (male or female):"
        except ValueError as e:
            return str(e)
    
    if state["gender"] is None:
        state["gender"] = query.strip().lower()
        if state["gender"] not in ['male', 'female']:
            return "Invalid gender. Please provide 'male' or 'female':"
        return "Please provide the location and likelihood percentage (e.g., 'CityA with a likelihood percentage of 0.7'):"
    
    # Parse location and likelihood input
    if "with a likelihood percentage of" in query:
        location, likelihood_str = query.split("with a likelihood percentage of")
        location = location.strip()
        likelihood = float(likelihood_str.strip())
        state["location_likelihood_map"][location] = likelihood
        return "Do you want to add another location? Type 'yes' to add another location or 'no' to proceed with prediction:"
    
    # Handle "yes" or "no" input for adding another location
    if query.strip().lower() == "yes":
        return "Please provide the next location and likelihood percentage (e.g., 'CityB with a likelihood percentage of 0.6'):"
    
    if query.strip().lower() == "no":
        results_df = predict_likelihood(state["age"], state["gender"], state["location_likelihood_map"])
        
        # Reset state after prediction
        reset_state()

        # Format the output to show results for all locations
        if not results_df.empty:
            output = "Based on the recent health behavior analysis:\n\n"
            for _, row in results_df.iterrows():
                age_range = row['Age'] if row['Age'] != 'None' else 'the given age range'
                behavior = row['Predicted Behavior']
                condition = row['Condition']
                feedback = row['Feedback']
                output += (f"For location '{row['Location']}', it is likely that a {row['Gender']} in the age range '{age_range}' exhibits '{behavior}' behavior. "
                           f"Someone in this age range is most likely to {condition}. The feedback is '{feedback}'.\n\n")
            return output.strip()
        else:
            return "No results found."
    
    return "Invalid input. Please try again."

def start_prompt():
    """Initial prompt when starting the chatbot."""
    return "Welcome! What is your name?"

def reset_state():
    """Reset the state variables."""
    state.update({"age": None, "gender": None, "location_likelihood_map": {}, "name": None, "action": None})

 # Create the Gradio interface
iface = gr.Interface(fn=iterate_input, inputs="text", outputs="text", 
                     title="Health Behavior Prediction",
                     description="Enter your details to predict health behavior and receive personalized feedback.")
                     
# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()


# In[ ]:




