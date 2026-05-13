# Water Pipe Failure Prediction Analysis

The interactive web application dashboard for visualizing and analyzing water pipe failure Prediction analysis for a city such as melbourne. The dashboard helps maintenance teams to prioritize pipes using risk scores, rich visualizations with the AI-powered recommendations. 

## Features

- Interactive filters (Material, Pipe Age, Risk Level)
- Risk overview with key metrics and charts
- Detailed insights (Age distribution, Material risk, Age vs Length)
- Top high-risk pipes table with conditional formatting
- AI-powered maintenance reasoning - LLM integration 

## Project Structure

    ```bash
    MOP-Code
    └── Playground
        └── Project_3B_T126
            ├── dashboard
            │   ├── webapp.py
            │   ├── predict_wpf.py
            │   └── README.md
            └── data
                └── processed
                    └── melbourne_risk_llm_ready.csv
    ```

## Installation & Setup
- After cloing the project, navigate to the project directory 
- Use VScode (preferrably) or any other notebook editor

### Navigate to Project Folder

```bash
cd /path/to/the/project
```
## 1. Create and Activate Virtual Environment
- Create environment
    ```bash
    python -m venv venv

- Activate on Mac/Linux
    ```bash
    source venv/bin/activate

 - Activate on Windows
    ```bash
    venv\Scripts\activate

## 2. Install required Libraries
- Create a text file - requirements.txt that contain the following libraries:

        streamlit
        pandas
        plotly
        openai
        streamlit-folium
        folium

- Install the requirements.txt
     ```bash
     pip install -r requirements.txt
- Or direct installation: 
    ```bash
    pip install streamlit pandas plotly openai streamlit-folium folium

## 3. Configure API Key (AI Features)

- Create .streamlit/secrets.toml:
  
   [XAI] api_key = "xai-your-api-key-here"

## 4. Run the application
- Ensure you are in your project folder

  ```bash
   streamlit run webapp.py
