from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import requests
import json
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from flask_caching import Cache

app = Flask(__name__)

# Flask-Caching configuration
cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})

# API Configuration
DATASET_IDS = [
    "social-indicators-for-city-of-melbourne-residents-2023",
    "social-indicators-for-city-of-melbourne-residents-2022",
    "social-indicators-for-city-of-melbourne-residents-2021",
    "social-indicators-for-city-of-melbourne-residents-2020",
    "social-indicators-for-city-of-melbourne-residents-2019",
    "social-indicators-for-city-of-melbourne-residents-2018",
]
API_URL = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/{dataset_id}/exports/json"


def fetch_and_prepare_data():
    all_data = []
    for dataset_id in DATASET_IDS:
        response = requests.get(API_URL.format(dataset_id=dataset_id))
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            if "topic" in df.columns:
                df = df[df["topic"].str.contains("Food security", na=False)]
            if "year" in df.columns:
                df["year"] = df["year"].astype(str)  # Ensure year is string for filtering
            all_data.append(df)
        else:
            print(f"Failed to fetch data for {dataset_id}")

    all_df = pd.concat(all_data, ignore_index=True)

    # Combine respondent_group columns
    if "respondent_group0" in all_df.columns:
        all_df["respondent_group"] = all_df["respondent_group"].combine_first(all_df["respondent_group0"])
        all_df.drop(columns=["respondent_group0"], inplace=True)

    # Replace inconsistent values
    all_df["respondent_group"] = all_df["respondent_group"].replace({
        "Kensington/ Flemingon 3031": "Kensington / Flemington 3031",
        "Southbank/ South Wharf 3006": "South Wharf / Southbank 3006",
        "South Yarra 3141 / Melbourne (St Kilda Road) 3004": "South Yarra 3141 / Melbourne/St Kilda Road 3004",
    })

    # Add food insecurity labels
    all_df["food_insecurity"] = all_df["description"].map({
        "Ran out of food": "Ran out of food",
        "Skipped meals": "Skipped meals",
        "Worried food would run out": "Worried food would run out",
        "Experienced food insecurity (worried food would run out and/or skipped meals "
        "and/or ran out of food)": "Insecurity (multiple concerns)",
        "Experienced food insecurity (worried food would run out and/or skipped meals "
        "and/or ran out of food and/or accessed emergency food relief services)": "Insecurity (multiple concerns + relief)",
        "Accessed emergency food relief services": "Accessed food relief services",
    })

    # Classify Age, Suburb, and Gender
    all_df["Age"] = all_df["respondent_group"].apply(classify_age)
    all_df["Suburb"] = all_df["respondent_group"].apply(classify_suburb)
    all_df["Gender"] = all_df["respondent_group"].apply(classify_gender)

    return all_df


# Classification Functions
def classify_age(value):
    if pd.isna(value):  # Handle missing values
        return None
    value = value.strip().title()
    if "Years" in value:
        return value
    return None


def classify_suburb(value):
    if pd.isna(value):
        return None
    value = value.strip().title()
    if "Years" not in value and value not in ["Male", "Female"]:
        return value
    return None


def classify_gender(value):
    if pd.isna(value):
        return None
    value = value.strip().title()
    if value in ["Male", "Female"]:
        return value
    return None


# Cached function to load data
@cache.cached(timeout=3600, key_prefix="data_cache")
def get_cached_data():
    return fetch_and_prepare_data()

#homepage
@app.route("/")
def main_page():
    return render_template("index.html")

#foodsecurity distribution
@app.route("/pie-chart")
def pie_chart():
    data = get_cached_data()  # Load cached data
    description_counts = data["food_insecurity"].value_counts().reset_index()
    description_counts.columns = ["food_insecurity", "Count"]

    # Create pie chart
    fig = px.pie(
        description_counts,
        names="food_insecurity",
        values="Count",
        title="Distribution of Food Insecurity<br><sup>Based on data from Melbourne residents (2018-2023)</sup>",
        color_discrete_sequence=px.colors.sequential.Emrld
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>"
    )
    fig.update_layout(
        width=800, height=800,
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=50, b=0), template="plotly_white"
    )

    # Render HTML with the plot
    plot_html = fig.to_html(full_html=False)
    return render_template("foodSecurityDistribution.html", plot=plot_html)

#demographics visualisation
@app.route("/demographics-visualization", methods=["GET", "POST"])
def demographics_visualization():
    data = get_cached_data()  # Load cached data
    years = ["Select the Year"] + sorted(data["year"].dropna().unique()) + ["All Years"]
    categories = ["Select the Category", "Age", "Gender", "Suburb"]

    selected_year = request.form.get("year", "Select the Year")
    selected_category = request.form.get("category", "Select the Category")

    # Filter the data
    if selected_year != "Select the Year" and selected_year != "All Years":
        filtered_data = data[data["year"] == selected_year]
    elif selected_year == "All Years":
        filtered_data = data
    else:
        filtered_data = pd.DataFrame()

    if filtered_data.empty or selected_category == "Select the Category":
        return render_template(
            "demographics_visualization.html",
            years=years,
            categories=categories,
            selected_year=selected_year,
            selected_category=selected_category,
            error_message="Please select both a year and a category to view the distribution.",
            graph_json=None
        )

    counts = filtered_data[selected_category].value_counts().reset_index()
    counts.columns = [selected_category, "Count"]
    fig = px.bar(
        counts,
        x=selected_category,
        y="Count",
        title=f"{selected_category} Distribution ({selected_year})",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(template="plotly_white", xaxis_title=selected_category, yaxis_title="Count")
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    return render_template(
        "demographics_visualization.html",
        years=years,
        categories=categories,
        selected_year=selected_year,
        selected_category=selected_category,
        graph_json=graph_json
    )

#demographics by grouping food security
@app.route("/demographics-food-security")
def demographics_food_security():
    data = get_cached_data()  # Fetch the cached data
    
    # Melt the data for plotting
    melted_df = pd.melt(
        data,
        id_vars=['food_insecurity'],
        value_vars=['Age', 'Gender', 'Suburb'],
        var_name='Category',
        value_name='Category Value'
    ).dropna(subset=['Category Value'])

    # Group by food insecurity type, category, and demographic value
    description_counts = melted_df.groupby(
        ['food_insecurity', 'Category', 'Category Value']
    ).size().reset_index(name='Count')

    # Create the bar plot
    fig = px.bar(
        description_counts,
        x='Category Value',
        y='Count',
        color='food_insecurity',
        facet_row='Category',
        title="Distribution of Food Insecurity Across Demographics",
        labels={'Count': 'Number of Responses', 'Category Value': 'Category Subgroup'},
        category_orders={"Category": ["Age", "Gender", "Suburb"]}
    )

    # Customize layout
    fig.update_layout(
        height=1000,
        template="plotly_white",
        yaxis_title="Number of Responses",
        xaxis_title="Category Subgroup",
        legend_title="Food Insecurity Type",
    )

    # Add axis title to each facet row
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Render the HTML with the plot
    plot_html = fig.to_html(full_html=False)
    return render_template("demographicsFoodSecurity.html", plot=plot_html)

#trend analysis with category and their values
@app.route("/trend-analysis", methods=["GET", "POST"])
def trend_analysis():
    data = get_cached_data()  # Load cached data
    
    # Ensure the year column is numeric
    data["year"] = pd.to_numeric(data["year"], errors="coerce")

    # Define categories for the dropdown
    categories = ["Select Category", "Age", "Gender", "Suburb"]
    selected_category = request.form.get("category", "Select Category")
    selected_value = request.form.get("value", "Select Value")

    # Populate specific values for the selected category
    values = ["Select Value"]
    if selected_category != "Select Category":
        values += sorted(data[selected_category].dropna().unique())

    # Filter data if valid category and value are selected
    filtered_data = data[
        (data[selected_category] == selected_value)
        if selected_category != "Select Category" and selected_value != "Select Value"
        else []
    ]

    plot_html = None
    if not filtered_data.empty:
        # Aggregate data by year
        aggregated_df = filtered_data.groupby("year", as_index=False).agg({"result": "mean"})

        # Prepare data for prediction
        X = aggregated_df[["year"]].values
        y = aggregated_df["result"].values

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict future years
        future_years = np.array([[2024], [2025]])
        future_predictions = model.predict(future_years)

        # Create the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=aggregated_df["year"],
            y=aggregated_df["result"],
            mode="lines+markers",
            name="Historical Data",
            line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=future_years.flatten(),
            y=future_predictions,
            mode="markers+text",
            name="Predictions",
            marker=dict(color="red", size=10),
            text=[f"{pred:.2f}%" for pred in future_predictions],
            textposition="top center"
        ))
        fig.update_layout(
            title=f"Food Security Trend for {selected_value} ({selected_category})",
            xaxis_title="Year",
            yaxis_title="Percentage",
            template="plotly_white",
            legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h"),
            height=600
        )
        plot_html = fig.to_html(full_html=False)

    return render_template(
        "trendAnalysis.html",
        categories=categories,
        values=values,
        selected_category=selected_category,
        selected_value=selected_value,
        plot_html=plot_html
    )

#combined trends with category and all their values
@app.route("/combined-trends", methods=["GET", "POST"])
def combined_trends():
    data = get_cached_data()  # Load cached data

    # Define categories for the dropdown
    categories = ["Select Category", "Age", "Gender", "Suburb"]
    selected_category = request.form.get("category", "Select Category")

    # Populate unique values for the selected category
    unique_values = []
    if selected_category != "Select Category":
        unique_values = sorted(data[selected_category].dropna().unique())

    plot_html = None
    if unique_values:
        # Create Plotly Figure
        fig = go.Figure()

        # Define color palettes for historical and predicted data
        historical_colors = px.colors.qualitative.Plotly
        predicted_colors = px.colors.qualitative.Pastel

        for idx, value in enumerate(unique_values):
            filtered_df = data[data[selected_category] == value]
            if filtered_df.empty:
                continue

            # Aggregate data by year
            aggregated_df = filtered_df.groupby("year", as_index=False).agg({"result": "mean"})

            # Prepare data for linear regression
            X = aggregated_df[["year"]].values
            y = aggregated_df["result"].values

            # Train a linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict for future years
            future_years = np.array([[2024], [2025]])
            future_predictions = model.predict(future_years)

            # Add historical data to the plot
            fig.add_trace(go.Scatter(
                x=aggregated_df["year"],
                y=aggregated_df["result"],
                mode="lines+markers",
                name=f"{value} (Historical)",
                line=dict(color=historical_colors[idx % len(historical_colors)], dash="solid"),
                marker=dict(size=8),
                hovertemplate=f"<b>{value}</b><br>Year: %{{x}}<br>Result: %{{y:.2f}}%<extra></extra>"
            ))

            # Add predictions to the plot
            fig.add_trace(go.Scatter(
                x=future_years.flatten(),
                y=future_predictions,
                mode="lines+markers",
                name=f"{value} (Predicted)",
                line=dict(color=predicted_colors[idx % len(predicted_colors)], dash="dot"),
                marker=dict(size=8),
                hovertemplate=f"<b>{value}</b><br>Year: %{{x}}<br>Prediction: %{{y:.2f}}%<extra></extra>"
            ))

        # Customize layout
        fig.update_layout(
            title=f"Combined Trends by {selected_category}",
            xaxis_title="Year",
            yaxis_title="Percentage",
            template="plotly_white",
            height=650,
            width=1300,
            legend=dict(
                x=1.05, y=1, orientation="v", title_text="", font=dict(size=10)
            ),
            margin=dict(l=40, r=250, t=50, b=50),
            hovermode="x unified"
        )

        # Convert the figure to HTML
        plot_html = fig.to_html(full_html=False)

    # Render the dropdown and plot
    return render_template(
        "combinedTrends.html",
        categories=categories,
        selected_category=selected_category,
        plot_html=plot_html
    )

#trends of foos security types across the demographics
@app.route("/food-insecurity-trends", methods=["GET", "POST"])
def food_insecurity_trends():
    data = get_cached_data()  # Load cached data

    # Prepare dropdown options
    category_options = ["Select Category", "Age", "Gender", "Suburb"]
    food_insecurity_options = ["Select the type"] + data["food_insecurity"].dropna().unique().tolist()

    selected_category = request.form.get("category", "Select Category")
    selected_food_insecurity = request.form.get("food_insecurity", "Select the type")

    plot_html = None
    if selected_category != "Select Category" and selected_food_insecurity != "Select the type":
        # Filter data
        filtered_data = data[
            (data["food_insecurity"] == selected_food_insecurity) & 
            (data[selected_category].notnull())
        ]

        if not filtered_data.empty:
            unique_values = filtered_data[selected_category].dropna().unique()
            fig = go.Figure()

            # Generate the plot for each unique value
            for value in unique_values:
                category_filtered_data = filtered_data[filtered_data[selected_category] == value]

                # Aggregate data by year
                aggregated_data = category_filtered_data.groupby("year", as_index=False).agg({"result": "mean"})

                # Prepare data for linear regression
                X = aggregated_data[["year"]].values
                y = aggregated_data["result"].values

                # Train linear regression model
                model = LinearRegression()
                model.fit(X, y)

                # Predict future years
                future_years = np.array([[2024], [2025]])
                future_predictions = model.predict(future_years)

                # Add historical data to the plot
                fig.add_trace(go.Scatter(
                    x=aggregated_data["year"],
                    y=aggregated_data["result"],
                    mode="lines+markers",
                    name=f"{value} (Historical)",
                    line=dict(dash="solid")
                ))

                # Add predictions to the plot
                fig.add_trace(go.Scatter(
                    x=future_years.flatten(),
                    y=[round(pred, 2) for pred in future_predictions],
                    mode="lines+markers",
                    name=f"{value} (Predicted)",
                    line=dict(dash="dot"),
                ))

            # Update layout
            fig.update_layout(
                title=f"Trends by {selected_category} for '{selected_food_insecurity}'",
                xaxis_title="Year",
                yaxis_title="Percentage",
                template="plotly_white",
                height=600,
                legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h"),
                hovermode="x unified",
            )

            # Generate Plotly HTML
            plot_html = fig.to_html(full_html=False)

    return render_template(
        "food_insecurity_trends.html",
        category_options=category_options,
        food_insecurity_options=food_insecurity_options,
        selected_category=selected_category,
        selected_food_insecurity=selected_food_insecurity,
        plot_html=plot_html,
    )

if __name__ == "__main__":
    app.run(debug=True)

# If we head to http://127.0.0.1:5000 (since we have not mentioned any port, it defaults to port 5000), our home page will appear.
# If it doesn't run on port 5000, change it to an available port using the following: app.run(debug=True, port=<available_port>)
# From the home page, we can navigate to other functionalities, which are linked to separate HTML pages.
# These other pages are accessible via buttons on our home page.



