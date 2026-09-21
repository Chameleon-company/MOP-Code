# HOUSING CHATBOT

## Overview

The housing chatbot aims to recommend the most suitable properties for the users based on their preferences. Besides basic information such as property type (House, Apartment), rental fee, number of bedrooms/bathrooms, etc, it can handle queries that include distance and travelling time from the properties to other places (such as Deakin University), in both driving and public transportation mode, which listing sites such as [realestate.com.au](realestate.com.au) currently do not support. In trimester 2 - 2025, our team has built the chatbot that could recommend properties based on a static crawled listing, it can handle multi-turn conversations where the user query is too vague (e.g. "I want to find a room") so that it could understand more about user's preferences. Based on the preferences, including basic information and distance/time to other places, the chatbot uses a simple ranking model with linear regression to rank the properties. The chatbot is built with LangGraph, with Gemini Flash 2.0 as the core LLM. The UI is build with Chainlit.

**Flowchat**:

![Flow chart of the chatbot](./docs/Housing%20chatbot%20flowchart.png)

## Code structure

```
. Housing chatbot/
├── chatbot/
│   └── graph.py # Core components of the chatbot (compiled in LangGraph)
├── configs/
│   ├── json_schema.py # Pre-defined housing schema to help the LLM extract housing entities from user's query.
│   └── prompts.py # System prompt for the LLM to extract housing entities.
├── data
├── docs
├── public # Logo files for chainlit
├── tests # Some test cases
├── utils/
│   └── utils # Data processing pipelines
├── .env
├── app.py # Chainlit app
├── get_data.py # To download GTFS data
└── runner.py # To run the chatbot on terminal
```

## How to run the code

### 1. Setup the environment
- `Python >=3.11`.
- `pip install langgraph langchain-google-genai dotenv chainlit geopy scipy openrouteservice pandas folium geopandas`.
- Add your Gemini API key and GTFS API key in the `.env` file.

### 2. Download GTFS data (for the first time only)
`python get_data.py`

### 3. Run the code on terminal (for testing purposes)
`python runner.py`

### 4. Run on UI
`chainlit run app.py -w`

## Future development
Future plans for the chatbot might include: 
1. Live listing data through API or crawling pipelines; 
2. Full distance/travelling time for public transportation, including trams and buses (Right now, it only handles walking, driving, and travelling by trains);
3. Enable for buying use case (it's only built for rental right now).
