from flask import Flask, render_template, request, jsonify
import health_safety 
import subjective_wellbeing
from map import create_map
from dash_app import create_dash_app
import pandas as pd

app = Flask(__name__)

#-------------------------
#Routes for the main pages

#Home page
@app.route('/')
def home():
    return render_template('home.html')

#Chatbot
@app.route('/chatbot_main')
def chatbot_main():
    return render_template('chatbot_main.html')

#Vehicle
@app.route('/vehicle_main')
def vehicle_main():
    return render_template('vehicle_main.html')

#Traffic
@app.route('/traffic_main')
def traffic_main():
    return render_template('traffic_main.html')

#Health
@app.route('/health_main')
def health_main():
    return render_template('health_main.html')

#-------------------------
#Routes for the Health behaviour project pages
#Health Safety
@app.route('/health_safety', methods=['GET', 'POST'])
def health_safety_app():
    return health_safety.index()

@app.route('/health_safety/predict', methods=['POST'])
def health_data_view():
    category = request.form.get('category')
    gender = request.form.get('gender')
    age_range = request.form.get('age_range')
    suburb = request.form.get('suburb')

    # Pass the data to the health_safety.predict function to handle the logic and get the response
    health_data = health_safety.predict(category, gender, age_range, suburb)

    # Return the prediction data as JSON
    return jsonify(health_data)

######################################################################################################
# Subjective Wellbeing

# Routes for Subjective Wellbeing
# Data source
data_path = r"D:\MOP-Code-mtortely\MOP-Code-mtortely\artificial-intelligence\AI Flask webapp - Student Projects\subjective_wellbeing_cleaned.csv"

# Subjective Wellbeing Home Page
@app.route('/subjective_wellbeing')
def subjective_wellbeing_home():
    return subjective_wellbeing.home()

# Subjective Wellbeing Data Page
@app.route('/subjective_wellbeing_data')
def subjective_wellbeing_data():
    return render_template('data.html')

dash_app = create_dash_app(app)

@app.route('/map', methods=['GET', 'POST'])
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

@app.route('/data')
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

#-------------------------
# Running the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Set to True for development, False for production
