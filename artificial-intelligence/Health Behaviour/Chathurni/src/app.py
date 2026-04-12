from flask import Flask, render_template, request, redirect, url_for
from dash_app import create_dash_app
from map import create_map
import pandas as pd

# Initialize Flask app
flask_app = Flask(__name__)
create_dash_app(flask_app)

# Data source
data_path = "D:/Chathurni/src/data/processed/subjective_wellbeing_cleaned.csv"

@flask_app.route('/')
def home():
    return render_template('index.html')

@flask_app.route('/search')
def search():
    """
    Handle search queries and redirect to the relevant pages based on the query.
    """
    query = request.args.get('query', '').lower()
    if not query:
        return render_template('search.html', results=[], query=query)
    
    # Map queries to specific pages
    if 'dashboard' in query:
        return redirect(url_for('dashboard'))  # Redirect to /dashboard/
    elif 'map' in query or 'interactive map' in query:
        return redirect(url_for('map_view'))  # Redirect to /map
    elif 'data' in query or 'dataset' in query:
        return redirect(url_for('data_view'))  # Redirect to /data
    
    # Fallback: Search within dataset
    df = pd.read_csv(data_path)
    results = df[df.apply(
        lambda row: query in str(row['Subtopics']).lower() or
                    query in str(row['Category']).lower() or
                    query in str(row['year']).lower(), axis=1
    )]
    
    return render_template(
        'search.html',
        query=query,
        results=results.to_dict(orient='records'),
        column_names=results.columns.tolist() if not results.empty else []
    )

@flask_app.route('/dashboard/')
def dashboard():
    return redirect('/dashboard/')  

@flask_app.route('/map', methods=['GET', 'POST'])
def map_view():
    subtopic = request.args.get('subtopic', 'All')
    year = request.args.get('year', 'All')
    folium_map = create_map(subtopic=subtopic, year=year)
    map_html = folium_map._repr_html_()

    df = pd.read_csv(data_path)
    subtopics = sorted(df['Subtopics'].unique())
    years = sorted(df['year'].unique())

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
    df = pd.read_csv(data_path)
    subtopic = request.args.get('subtopic', 'All')
    category = request.args.get('category', 'All')
    year = request.args.get('year', 'All')

    filtered_df = df.copy()
    if subtopic != 'All':
        filtered_df = filtered_df[filtered_df['Subtopics'] == subtopic]
    if category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == category]
    if year != 'All':
        filtered_df = filtered_df[filtered_df['year'] == int(year)]

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
    import os
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    flask_app.run(debug=debug_mode, port=2022)
