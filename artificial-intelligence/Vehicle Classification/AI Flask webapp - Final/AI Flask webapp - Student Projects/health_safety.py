import pandas as pd
from flask import Flask, render_template, request, jsonify
import plotly.express as px
import plotly.io as pio
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Read the dataset
df = pd.read_csv('SafetyPerception_data.csv')

# Get unique categories for filters
genders = df[df['Category'] == 'Gender']['Subcategory'].unique()
age_ranges = df[df['Category'] == 'AgeRange']['Subcategory'].unique()
suburbs = df[df['Category'] == 'Suburb']['Subcategory'].unique()

@app.route('/health_safety')
def index():
    # Render the initial form with available filters
    return render_template('health_safety.html', genders=genders, age_ranges=age_ranges, suburbs=suburbs)

@app.route('/health_safety/predict/', methods=['POST'])
def predict(category, gender, age_range, suburb):

    selected_data_day = []
    selected_data_night = []

    if category == 'neighbourhood':
        description_day = 'neighbourhoodDay'
        description_night = 'neighbourhoodNight'
    else:
        description_day = 'transportDay'
        description_night = 'transportNight'

    day_data = df[df['Description'] == description_day]
    night_data = df[df['Description'] == description_night]

    if gender != 'Select a Gender':
        day_gender = day_data[day_data['Subcategory'] == gender]
        night_gender = night_data[night_data['Subcategory'] == gender]
        selected_data_day.append(day_gender)
        selected_data_night.append(night_gender)

    if age_range != 'Select an Age Range':
        day_age = day_data[day_data['Subcategory'] == age_range]
        night_age = night_data[night_data['Subcategory'] == age_range]
        selected_data_day.append(day_age)
        selected_data_night.append(night_age)

    if suburb != 'Select a Suburb':
        day_sub = day_data[day_data['Subcategory'] == suburb]
        night_sub = night_data[night_data['Subcategory'] == suburb]
        selected_data_day.append(day_sub)
        selected_data_night.append(night_sub)

    if not selected_data_day or not selected_data_night:
        return jsonify({'error': 'No data available for the selected filters'})


    combo_data_day = pd.concat(selected_data_day, ignore_index=True)
    combo_data_night = pd.concat(selected_data_night, ignore_index=True)
    
    combo_data_day = combo_data_day.groupby(['Year']).agg({'Percentage': 'mean'}).reset_index()
    combo_data_night = combo_data_night.groupby(['Year']).agg({'Percentage': 'mean'}).reset_index()

    # Prediction (Linear regression model for demonstration)
    X_day = combo_data_day[['Year']]
    y_day = combo_data_day['Percentage']
    model = LinearRegression()
    model.fit(X_day, y_day)
    predicted_percentages_day = model.predict(pd.DataFrame({'Year': [2024, 2025]}))

    X_night = combo_data_night[['Year']]
    y_night = combo_data_night['Percentage']
    model = LinearRegression()
    model.fit(X_night, y_night)
    predicted_percentages_night = model.predict(pd.DataFrame({'Year': [2024, 2025]}))

    # prediction_text = f"Predicted percentage of {gender} aged {age_range} in {suburb} feeling safe:\n" \
    #                   f"2024 Day: {predicted_percentages_day[0]:.2f}% | 2025 Day: {predicted_percentages_day[1]:.2f}%\n" \
    #                   f"2024 Night: {predicted_percentages_night[0]:.2f}% | 2025 Night: {predicted_percentages_night[1]:.2f}%"
    
    # Create line graphs for Day and Night
    fig_day = px.line(combo_data_day, x='Year', y='Percentage', title=f"Feeling Safe During the Day: {category.capitalize()}")
    fig_day.add_scatter(x=[2024, 2025], y=predicted_percentages_day, mode='markers', name='Predictions', marker=dict(color='red'))
    fig_night = px.line(combo_data_night, x='Year', y='Percentage', title=f"Feeling Safe at Night: {category.capitalize()}")
    fig_night.add_scatter(x=[2024, 2025], y=predicted_percentages_night, mode='markers', name='Predictions', marker=dict(color='red'))

    fig_day.update_layout(
        paper_bgcolor= '#A3E4F7',
        showlegend = False,
        font = dict(color='#000'),
        yaxis=dict(
            range=[0, 100]
        )
    )
    fig_night.update_layout(
        paper_bgcolor= '#003C55',
        showlegend = False,
        font = dict(color='#fff'),
        yaxis=dict(
            range=[0, 100]
        )
    )

    # Convert figures to JSON
    fig_json_day = pio.to_json(fig_day)
    fig_json_night = pio.to_json(fig_night)

    # Return data including predictions and graphs as JSON
    return {
        'graph_day': fig_json_day,
        'graph_night': fig_json_night,
    }

if __name__ == '__main__':
    app.run(debug=True)