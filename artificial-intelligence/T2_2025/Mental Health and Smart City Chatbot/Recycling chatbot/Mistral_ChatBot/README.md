## Project Overview

This chatbot helps users ask questions about recycling rules in Victoria by providing contextual answers based on a dataset of recycling instructions combined with AI-generated explanations from the Mistral API.

---

## Tech Stack

- **Backend:** Python, Flask, Flask-CORS  
- **Frontend:** React  
- **AI Model:** Mistral API (https://api.mistral.ai)  
- **Dataset:** JSON file scraped from Victoria recycling resources  

---

## Features

- Matches user queries to relevant recycling items from a local dataset  
- Constructs detailed prompts combining user input and dataset info  
- Uses Mistral API for generating conversational AI answers  
- React-based chat UI with real-time interaction and typing indicators  

---

## Dataset

The dataset (`recycling_data_clean.json`) contains detailed information about recycling items, including:

- Item names and common aliases  
- Category (e.g., soft plastics, chemicals, garden waste)  
- Bin types and colors  
- Recycling instructions  
- Drop-off locations with names and websites  
- Extra notes and sources  

The dataset is included in the repository and is based on official Victoria recycling guidelines.

---
