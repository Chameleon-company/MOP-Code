import pandas as pd
import requests
from io import StringIO

# Data Cleaning and Transformation
def perceptionofsafety_data(dataframe):
    filtered_df = dataframe[dataframe['topic'] == 'Perceptions of safety']

    # Step 2: Drop unnecessary columns if they exist
    drop_columns = ['indicator', 'topic', 'type', 'response', 'format', 'sample_size']
    filtered_df.drop(columns=[col for col in drop_columns if col in filtered_df.columns], inplace=True, errors='ignore')

    # Step 3: Add 'Category' column based on respondent grouping
    category_map = {
        'Male': 'Gender', 'Female': 'Gender',
        '18-24 years': 'AgeRange', '25-34 years': 'AgeRange', '35-44 years': 'AgeRange',
        '45-54 years': 'AgeRange', '55-64 years': 'AgeRange', '65+ years': 'AgeRange',
        '18-24': 'AgeRange', '25-34': 'AgeRange', '35-44': 'AgeRange',
        '45-54': 'AgeRange', '55-64': 'AgeRange', '65+': 'AgeRange'
    }
    filtered_df['Category'] = filtered_df['respondent_group'].map(category_map).fillna('Suburb')

    # Step 4: Reorganize columns and rename them
    reorganized_df = filtered_df[['description', 'Category', 'respondent_group', 'year', 'result']]
    reorganized_df.rename(columns={'description': 'Description', 'respondent_group': 'Subcategory', 'year' : 'Year', 'result': 'Percentage'}, inplace=True)

    # Clean up the 'Subcategory' column
    subcategory_replacements = {
        '18-24': '18-24 years', '24-34': '24-34 years', '25-34': '25-34 years', '35-44': '35-44 years',
        '45-54': '45-54 years', '55-64': '55-64 years', '65+': '65+ years',
        'Kensington/ Flemingon 3031': 'Kensington / Flemington 3031',
        'South Yarra 3141 / Melbourne (St Kilda Road) 3004': 'South Yarra 3141 / Melbourne/St Kilda Road 3004',
        'Southbank/ South Wharf 3006': 'South Wharf / Southbank 3006'
    }

    description_replacements = {
        'Feel safe by yourself in your neighbourhood - during the day' : 'neighbourhoodDay',
        'Feel safe by yourself in your neighbourhood - at night' : 'neighbourhoodNight',
        'Feel safe by yourself on public transport in and around City of Melbourne - at night' : 'transportDay',
        'Feel safe by yourself on public transport in and around City of Melbourne - during the day' : 'transportNight'
    }
    reorganized_df['Subcategory'].replace(subcategory_replacements, inplace=True)
    reorganized_df['Description'].replace(description_replacements, inplace=True)

    # Adjust the index to start from 1
    reorganized_df.index += 1

    return reorganized_df

# Fetch and Combine Datasets
def fetch_and_combine_datasets(dataset_ids, base_url):
    dataframes = []
    for dataset_id in dataset_ids:
        url = f'{base_url}{dataset_id}/exports/csv'
        try:
            response = requests.get(url)
            response.raise_for_status()
            # Load CSV content directly into DataFrame
            df = pd.read_csv(StringIO(response.content.decode('utf-8')), delimiter=';', on_bad_lines='skip')

            # Standardize column names
            df.rename(columns={'respondent_group0': 'respondent_group'}, inplace=True, errors='ignore')
            dataframes.append(df)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching dataset {dataset_id}: {e}")

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

# Main Execution Block
if __name__ == "__main__":
    BASE_URL = 'https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/'
    DATASET_IDS = [
        'social-indicators-for-city-of-melbourne-residents-2023',
        'social-indicators-for-city-of-melbourne-residents-2022',
        'social-indicators-for-city-of-melbourne-residents-2021',
        'social-indicators-for-city-of-melbourne-residents-2020',
        'social-indicators-for-city-of-melbourne-residents-2019',
        'social-indicators-for-city-of-melbourne-residents-2018'
    ]

    # Fetch and Combine Datasets
    combined_data = fetch_and_combine_datasets(DATASET_IDS, BASE_URL)

    if not combined_data.empty:
        # Process the combined data
        processed_data = perceptionofsafety_data(combined_data)

        # Save to CSV
        processed_data.to_csv('C:/Users/kaimo/Documents/UNI/- SIT378/PerceptionsOfSafety/SafetyPerception_data.csv', index=False)
        print("Data saved successfully.")
    else:
        print("Error collecting data.")
