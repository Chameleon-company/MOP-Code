import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go

def create_dash_app(flask_app):
    # Load the processed dataset
    data_path = "src/data/processed/subjective_wellbeing_cleaned.csv"
    processed_data = pd.read_csv(data_path)

    # Initialize Dash app with Flask server and Bootstrap dark theme
    dash_app = dash.Dash(
        __name__,
        server=flask_app,
        url_base_pathname='/dashboard/',
        external_stylesheets=[dbc.themes.DARKLY]
    )

    # Prepare options for dropdown menus
    subtopic_options = [{"label": sub, "value": sub} for sub in processed_data['Subtopics'].unique()]
    age_group_options = [{"label": sub, "value": sub} for sub in processed_data[processed_data['Category'] == 'Age Group']['Subcategory'].unique()]
    gender_options = [{"label": sub, "value": sub} for sub in processed_data[processed_data['Category'] == 'Gender']['Subcategory'].unique()]
    suburb_options = [{"label": sub, "value": sub} for sub in processed_data[processed_data['Category'] == 'Suburb']['Subcategory'].unique()]

    # Define Dash app layout
    dash_app.layout = dbc.Container(
        [
            # Title section
            dbc.Row(
                dbc.Col(
                    html.H1(
                        "Subjective Wellbeing Dashboard",
                        className="text-center",
                        style={"color": "#08af64", "margin-bottom": "20px"}
                    )
                )
            ),
            # Filters section
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select Subtopic:", style={"color": "#ffffff", "font-weight": "bold"}),
                            dcc.Dropdown(
                                id='subtopic-dropdown',
                                options=subtopic_options,
                                value=processed_data['Subtopics'].unique()[0],
                                className="mb-3 bg-dark text-light"
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Select Age Group:", style={"color": "#ffffff", "font-weight": "bold"}),
                            dcc.Dropdown(
                                id='age-group-dropdown',
                                options=age_group_options,
                                value=processed_data[processed_data['Category'] == 'Age Group']['Subcategory'].unique()[0],
                                className="mb-3 bg-dark text-light"
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Select Gender:", style={"color": "#ffffff", "font-weight": "bold"}),
                            dcc.Dropdown(
                                id='gender-dropdown',
                                options=gender_options,
                                value=processed_data[processed_data['Category'] == 'Gender']['Subcategory'].unique()[0],
                                className="mb-3 bg-dark text-light"
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Select Suburb:", style={"color": "#ffffff", "font-weight": "bold"}),
                            dcc.Dropdown(
                                id='suburb-dropdown',
                                options=suburb_options,
                                value=processed_data[processed_data['Category'] == 'Suburb']['Subcategory'].unique()[0],
                                className="mb-3 bg-dark text-light"
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-4",
            ),
            # Graph section
            dbc.Row(
                dbc.Col(
                    dcc.Graph(id='trend-graph'),
                    width=12,
                )
            ),
        ],
        fluid=True,
        className="bg-dark text-light py-4"
    )

    # Callback to update the graph based on user selections
    @dash_app.callback(
        Output('trend-graph', 'figure'),
        [
            Input('subtopic-dropdown', 'value'),
            Input('age-group-dropdown', 'value'),
            Input('gender-dropdown', 'value'),
            Input('suburb-dropdown', 'value')
        ]
    )
    def update_graph(selected_subtopic, selected_age_group, selected_gender, selected_suburb):
        # Filter data based on selected dropdown values
        filtered_data = processed_data[
            (processed_data['Subtopics'] == selected_subtopic) & (
                ((processed_data['Category'] == 'Age Group') & (processed_data['Subcategory'] == selected_age_group)) |
                ((processed_data['Category'] == 'Gender') & (processed_data['Subcategory'] == selected_gender)) |
                ((processed_data['Category'] == 'Suburb') & (processed_data['Subcategory'] == selected_suburb))
            )
        ]

        if filtered_data.empty:
            return go.Figure()

        # Prepare historical data
        historical_data = filtered_data[filtered_data['year'] <= 2023]
        years = historical_data['year'].values
        percentages = historical_data['Percentage'].values

        # Generate predictions for 2024 and 2025 using linear regression
        if len(years) > 1:
            slope, intercept = np.polyfit(years, percentages, 1)
            pred_2024 = slope * 2024 + intercept
            pred_2025 = slope * 2025 + intercept
        else:
            pred_2024, pred_2025 = None, None

        # Create the plotly figure
        figure = go.Figure()

        # Add historical data to the graph
        figure.add_trace(go.Scatter(
            x=historical_data['year'],
            y=historical_data['Percentage'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#0000FF'),
        ))

        # Add predictions to the graph
        if pred_2024 and pred_2025:
            figure.add_trace(go.Scatter(
                x=[2024, 2025],
                y=[pred_2024, pred_2025],
                mode='markers+text',
                name='Predictions',
                text=[f"{pred_2024:.2f}%", f"{pred_2025:.2f}%"],
                textposition="top center",
                marker=dict(color='red', size=10),
            ))

        # Update figure layout
        figure.update_layout(
            title=f"Trend for {selected_subtopic} ({selected_age_group}, {selected_gender}, {selected_suburb})",
            title_font=dict(size=16, color='#08af64'),
            xaxis_title="Year",
            yaxis_title="Percentage",
            legend=dict(x=0, y=-0.2, orientation="h"),
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor="#121212",
            plot_bgcolor="#1e1e1e",
            font=dict(color="#e0e0e0")
        )

        return figure

    return dash_app
