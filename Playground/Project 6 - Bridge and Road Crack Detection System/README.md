# SIT378 Team Project - Road Crack Detection Team

## Project Overview
This is a Bridge and Road Crack Detection System with 4 main components:
1. Backend API (`app.py`) that connects 
   - Crack Detection Computer Vision module (`crack_detection/`)
   - LLM Pipeline (`LLM_pipeline`) for report generation
2. React Dashboard (`Dashboard/`) for frontend

## Set up
### Prerequisites
- Python 3.10+
- Node.js (for the Dashboard)
- An OpenAI API key
- A Supabase account/project

### Set up environment variables
Export these to your environment
```
SUPABASE_URL=your_supabase_url
SUPABASE_PUBLISHABLE_DEFAULT_KEY=your_supabase_key
OPENAI_API_KEY=your_openai_key
```

### Install Python dependencies
`pip install -r requirements.txt`

### Set up supebase
Create a table named `crack_reports` with the following columns
|        Column        |  Type  |
|:--------------------:|:------:|
| id                   | uuid   |
| imageid              | text   |
| severity             | text   |
| numcracks            | int8   |
| crackarearatio       | float8 |
| estimatedcracklength | float8 |
| damagelevel          | float8 |
| reportstatus         | text   |
| riskassessment       | text   |
| repairactions        | text   |
| inspectionschedule   | text   |

Make sure to Disable RLS.

Go to Storage and create 2 buckets named `original-images` and `crack-masks`.

## Run application
Run the backend first
```
python api.py
```

Run the frontend
```
cd Dashboard
npm install
npm run dev
```
