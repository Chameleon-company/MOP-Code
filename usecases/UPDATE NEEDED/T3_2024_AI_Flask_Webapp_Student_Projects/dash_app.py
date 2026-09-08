import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go


def create_dash_app(flask_app):
    # Load the processed data
    data_path = r"D:\MOP-Code-mtortely\MOP-Code-mtortely\artificial-intelligence\AI Flask webapp - Student Projects\subjective_wellbeing_cleaned.csv"
    processed_data = pd.read_csv(data_path)

    # Initialize Dash app with Bootstrap theme
    dash_app = dash.Dash(
        __name__,
        server=flask_app,
        url_base_pathname='/dashboard/',
        external_stylesheets=[dbc.themes.BOOTSTRAP]  # Using Bootstrap for styling
    )

    # Dropdown options
    subtopic_options = [{"label": sub, "value": sub} for sub in processed_data['Subtopics'].unique()]
    age_group_options = [{"label": sub, "value": sub} for sub in processed_data[processed_data['Category'] == 'Age Group']['Subcategory'].unique()]
    gender_options = [{"label": sub, "value": sub} for sub in processed_data[processed_data['Category'] == 'Gender']['Subcategory'].unique()]
    suburb_options = [{"label": sub, "value": sub} for sub in processed_data[processed_data['Category'] == 'Suburb']['Subcategory'].unique()]

    # Dash layout
    dash_app.layout = dbc.Container(
        [
            # Back to Home button
            dbc.Row(
                dbc.Col(
                    html.A(
                        "← Back to Home",
                        href="/",
                        className="btn btn-link",
                        style={"fontSize": "18px", "marginBottom": "10px", "color": "#08af64", "textDecoration": "none"}
                    )
                )
            ),
            # Page Header
            dbc.Row(
                dbc.Col(
                    html.H1(
                        "Subjective Wellbeing Dashboard",
                        className="text-center mb-4",
                        style={"fontWeight": "bold", "color": "#08af64"}  # Green title
                    )
                )
            ),
            # Filters
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select Subtopic:", className="font-weight-bold"),
                            dcc.Dropdown(
                                id='subtopic-dropdown',
                                options=subtopic_options,
                                value=processed_data['Subtopics'].unique()[0],
                                className="mb-3",
                                style={"backgroundColor": "white"}
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Select Age Group:", className="font-weight-bold"),
                            dcc.Dropdown(
                                id='age-group-dropdown',
                                options=age_group_options,
                                value=processed_data[processed_data['Category'] == 'Age Group']['Subcategory'].unique()[0],
                                className="mb-3",
                                style={"backgroundColor": "white"}
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Select Gender:", className="font-weight-bold"),
                            dcc.Dropdown(
                                id='gender-dropdown',
                                options=gender_options,
                                value=processed_data[processed_data['Category'] == 'Gender']['Subcategory'].unique()[0],
                                className="mb-3",
                                style={"backgroundColor": "white"}
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Select Suburb:", className="font-weight-bold"),
                            dcc.Dropdown(
                                id='suburb-dropdown',
                                options=suburb_options,
                                value=processed_data[processed_data['Category'] == 'Suburb']['Subcategory'].unique()[0],
                                className="mb-3",
                                style={"backgroundColor": "white"}
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-4",
            ),
            # Graph
            dbc.Row(
                dbc.Col(
                    dcc.Graph(id='trend-graph', style={"backgroundColor": "white"}),
                    width=12,
                )
            ),
            # Predictions Section
            dbc.Row(
                dbc.Col(
                    html.Div(id='trend-description', className="mt-4", style={"fontSize": "16px", "color": "#2b2b2b"}),
                    width=12,
                )
            ),
            # Additional Insights
            dbc.Row(
                dbc.Col(
                    html.Div(id="predictions-details", className="mt-4", style={"fontSize": "14px", "color": "#4a4a4a"}),
                    width=12,
                )
            )
        ],
        fluid=True,
        style={"backgroundColor": "white", "padding": "20px"}
    )

    # Callback for interactivity
    @dash_app.callback(
        [
            Output('trend-graph', 'figure'),
            Output('trend-description', 'children'),
            Output('predictions-details', 'children'),
        ],
        [
            Input('subtopic-dropdown', 'value'),
            Input('age-group-dropdown', 'value'),
            Input('gender-dropdown', 'value'),
            Input('suburb-dropdown', 'value')
        ]
    )
    def update_graph(selected_subtopic, selected_age_group, selected_gender, selected_suburb):
        # Filter data based on dropdown selections
        filtered_data = processed_data[
            (processed_data['Subtopics'] == selected_subtopic) &
            (
                (processed_data['Category'] == 'Age Group') & (processed_data['Subcategory'] == selected_age_group) |
                (processed_data['Category'] == 'Gender') & (processed_data['Subcategory'] == selected_gender) |
                (processed_data['Category'] == 'Suburb') & (processed_data['Subcategory'] == selected_suburb)
            )
        ]

        if filtered_data.empty:
            return go.Figure(), "No data available for the selected criteria.", ""

        # Ensure data types are correct
        filtered_data['year'] = pd.to_numeric(filtered_data['year'], errors='coerce').dropna()
        filtered_data['Percentage'] = pd.to_numeric(filtered_data['Percentage'], errors='coerce').dropna()

        # Sort data by year
        filtered_data = filtered_data.sort_values(by='year')

        # Prepare historical data for the graph
        historical_data = filtered_data[filtered_data['year'] <= 2023]

        # Interpolate missing years
        all_years = pd.DataFrame({'year': range(historical_data['year'].min(), historical_data['year'].max() + 1)})
        historical_data = pd.merge(all_years, historical_data, on='year', how='left')
        historical_data['Percentage'] = historical_data['Percentage'].interpolate()

        years = historical_data['year'].values
        percentages = historical_data['Percentage'].values

        # Predictions for 2024 and 2025 using linear regression
        if len(years) > 1:  
            slope, intercept = np.polyfit(years, percentages, 1)
            pred_2024 = slope * 2024 + intercept
            pred_2025 = slope * 2025 + intercept
        else:
            pred_2024, pred_2025 = None, None

        # Create the plotly figure
        figure = go.Figure()

        # Add historical data with lines and markers
        figure.add_trace(go.Scatter(
            x=historical_data['year'],
            y=historical_data['Percentage'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='green', width=3, dash='solid'),
            marker=dict(size=8, color='darkgreen'),
        ))

        # Add predictions
        if pred_2024 and pred_2025:
            figure.add_trace(go.Scatter(
                x=[historical_data['year'].iloc[-1], 2024, 2025],
                y=[historical_data['Percentage'].iloc[-1], pred_2024, pred_2025],
                mode='lines+markers',
                name='Predictions',
                line=dict(color='orange', width=2, dash='dot'),
                marker=dict(color='red', size=10),
            ))

        # Update layout
        figure.update_layout(
            title=f"Trend for {selected_subtopic} ({selected_age_group}, {selected_gender}, {selected_suburb})",
            xaxis_title="Year",
            yaxis_title="Percentage",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#2b2b2b"),
            legend=dict(x=0, y=-0.2, orientation="h"),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Generate trend description
        description = f"The trend for '{selected_subtopic}' in the '{selected_age_group}' age group, " \
                      f"gender '{selected_gender}', and suburb '{selected_suburb}' shows a percentage " \
                      f"change over the years {', '.join(map(str, years))}."

        # Predictions explanation
        predictions_text = ""
        if pred_2024 and pred_2025:
            predictions_text = f"<strong>Predictions:</strong><br>" \
                               f"2024: {pred_2024:.2f}%<br>" \
                               f"2025: {pred_2025:.2f}%<br>" \
                               f"The predicted trend shows {'growth' if pred_2025 > pred_2024 else 'decline'} over the next two years."

        return figure, description, predictions_text

    return dash_app
