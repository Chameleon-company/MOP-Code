import streamlit as st
import lightgbm as lgb
import pandas as pd

# Load your trained LightGBM model
lgbm_model = lgb.Booster(model_file='lgbm_model.txt')

# Streamlit app title
st.title("Melbourne House Price Prediction")

# Description of the app
st.write("""
This app predicts the **Melbourne House Price** based on selected features like the number of rooms, type of property, location, and more. Please input the details below to get the predicted price.

### Input Example:
- **Rooms:** Enter the total number of rooms (e.g., 1-10).
- **Type:** Select 'h' for House, 'u' for Unit, or 't' for Townhouse.
- **Distance:** Distance from Melbourne CBD in kilometers (e.g., 0.0 - 50.0).
- **Postcode:** Must be a valid 4-digit postcode (e.g., 3000 - 3999).
- **Bathroom:** Enter the number of bathrooms (e.g., 1-5).
- **Landsize:** Enter the land size in square meters (e.g., 50 - 1000).
- **Building Area:** Enter the total building area in square meters (e.g., 50 - 500).
- **Council Area:** Select the council area from the list provided.
""")

# User inputs for the selected features with validation
Rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3, help="Enter the number of rooms (1-10)")
Type = st.selectbox("Type of Property", ['h', 'u', 't'], help="Select the property type: 'h' for House, 'u' for Unit, 't' for Townhouse")
Distance = st.number_input("Distance from CBD (in km)", min_value=0.0, max_value=50.0, value=10.0, help="Enter the distance from Melbourne CBD (0.0 - 50.0)")
Postcode = st.number_input("Postcode", min_value=3000, max_value=3999, help="Enter a valid 4-digit postcode (e.g., 3000 - 3999)")
Bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2, help="Enter the number of bathrooms (1-5)")
Landsize = st.number_input("Landsize (in square meters)", min_value=50.0, max_value=1000.0, value=500.0, help="Enter the land size in square meters (50 - 1000)")
BuildingArea = st.number_input("Building Area (in square meters)", min_value=50.0, max_value=500.0, value=150.0, help="Enter the building area in square meters (50 - 500)")

# Full list of council areas
council_areas = [
    'Yarra City Council', 'Moonee Valley City Council', 'Port Phillip City Council', 'Darebin City Council',
    'Hobsons Bay City Council', 'Stonnington City Council', 'Boroondara City Council', 'Monash City Council',
    'Glen Eira City Council', 'Whitehorse City Council', 'Maribyrnong City Council', 'Bayside City Council',
    'Moreland City Council', 'Manningham City Council', 'Melbourne City Council', 'Banyule City Council',
    'Brimbank City Council', 'Kingston City Council', 'Hume City Council', 'Knox City Council', 'Maroondah City Council',
    'Casey City Council', 'Melton City Council', 'Greater Dandenong City Council', 'Nillumbik Shire Council',
    'Cardinia Shire Council', 'Whittlesea City Council', 'Frankston City Council', 'Macedon Ranges Shire Council',
    'Yarra Ranges Shire Council', 'Wyndham City Council', 'Moorabool Shire Council', 'Mitchell Shire Council'
]

CouncilArea = st.selectbox("Council Area", council_areas, help="Select the council area")

# Manually encode 'Type' and 'CouncilArea'
type_mapping = {'h': 0, 'u': 1, 't': 2}

# Create a dynamic mapping for council areas
council_area_mapping = {name: idx for idx, name in enumerate(council_areas)}

Type_encoded = type_mapping[Type]
CouncilArea_encoded = council_area_mapping[CouncilArea]

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'Rooms': [Rooms],
    'Type': [Type_encoded],
    'Distance': [Distance],
    'Postcode': [Postcode],
    'Bathroom': [Bathroom],
    'Landsize': [Landsize],
    'BuildingArea': [BuildingArea],
    'CouncilArea': [CouncilArea_encoded]
})

# Button to make prediction
if st.button("Predict"):
    # Make prediction
    prediction = lgbm_model.predict(input_data)
    
    # Display the prediction
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")
