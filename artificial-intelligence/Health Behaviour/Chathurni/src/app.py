from flask import Flask, render_template, request
from dash_app import create_dash_app
from map import create_map
import pandas as pd

# Initialize Flask app
flask_app = Flask(__name__)

# Integrate Dash app into Flask
create_dash_app(flask_app)

# CSV file containing the processed dataset
data_path = "D:/Chathurni/src/data/processed/subjective_wellbeing_cleaned.csv"

@flask_app.route('/')
def home():
    # Render the home page
    return render_template('index.html')

@flask_app.route('/map', methods=['GET', 'POST'])
def map_view():
    # Extract filters from query parameters
    subtopic = request.args.get('subtopic', 'All')
    year = request.args.get('year', 'All')

    # Generate the map using the provided filters
    folium_map = create_map(subtopic=subtopic, year=year)
    map_html = folium_map._repr_html_()

    # Load unique subtopics and years from the dataset
    df = pd.read_csv(data_path)
    subtopics = sorted(df['Subtopics'].unique())
    years = sorted(df['year'].unique())

    # Render the map page with the generated map and filter options
    return render_template(
        'map.html',
        map_html=map_html,
        subtopic=subtopic,
        year=year,
        subtopics=subtopics,
        years=years
    )

@flask_app.route('/data')
def data_view():
    # Load the dataset
    df = pd.read_csv(data_path)

    # Extract filter options from query parameters
    subtopic = request.args.get('subtopic', 'All')
    category = request.args.get('category', 'All')
    year = request.args.get('year', 'All')

    # Filter the dataset based on user selections
    filtered_df = df.copy()
    if subtopic != 'All':
        filtered_df = filtered_df[filtered_df['Subtopics'] == subtopic]
    if category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == category]
    if year != 'All':
        filtered_df = filtered_df[filtered_df['year'] == int(year)]

    # Render the data page with filtered data and dropdown options
    return render_template(
        'data.html',
        column_names=filtered_df.columns.tolist(),
        data=filtered_df.values.tolist(),
        subtopics=sorted(df['Subtopics'].unique()),
        years=sorted(df['year'].unique()),
        subtopic=subtopic,
        category=category,
        year=year
    )

if __name__ == '__main__':
    # Run the app in debug mode on port 2022
    flask_app.run(debug=True, port=2022)
