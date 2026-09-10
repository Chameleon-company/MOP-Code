# SIT378 Team Project - Road Crack Detection Team

## Project Overview
Video demo: https://www.youtube.com/watch?v=3JxI6ukVHXk  

Project made by: Long (s223128143@deakin.edu.au), Madhav (s223305696@deakin.edu.au), Callum (s224583534@deakin.edu.au) 


Infrastructure such as bridges, roads, and concrete structures develop cracks overtime due to environmental stress, aging materials, and heavy usage. Traditional inspection methods rely on manual visual assessment which can be time consuming, expensive, and prone to human error.
This project aims to develop an AI-powered infrastructure inspection assistant that automatically detects cracks in infrastructure images and generates maintenance recommendations. 

The system combines CV and LLM to produce engineering-style reports based on infrastructure maintenance guidelines. The system does the following:
1.	Accept infrastructure images with cracks (bridges, roads).
2.	Detect crack regions using segmentation models.
3.	Extract crack severity metrics from predicted crack masks.
4.	Retrieve relevant infrastructure maintenance guidelines using RAG.
5.	Generate an automated maintenance report using LLM.

In short, image -> crack detection -> crack metrics -> RAG -> LLM report. 



There are 4 main components:

1. Backend API (`api.py`) that connects
   - Crack Detection Computer Vision module (`crack_detection/`)
   - LLM Pipeline (`LLM_pipeline`) for report generation
2. Dashboard (`Dashboard/`) for React frontend, `app.py` for Streamlit frontend. 

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
| Column | Type |
|:--------------------:|:------:|
| id | uuid |
| imageid | text |
| severity | text |
| numcracks | int8 |
| crackarearatio | float8 |
| estimatedcracklength | float8 |
| damagelevel | float8 |
| reportstatus | text |
| riskassessment | text |
| repairactions | text |
| inspectionschedule | text |
| imageurl | text |
| crackmaskurl | text |
| overlayurl | text |

```sql
CREATE table crack_reports(
  id SERIAL PRIMARY KEY,
  imageid VARCHAR(300),
  severity VARCHAR(30),
  numcracks INT,
  crackarearatio FLOAT,
  estimatedcracklength INT,
  damagelevel FLOAT,
  reportstatus VARCHAR(20) NOT NULL DEFAULT 'None'
    CHECK (reportStatus IN ('None', 'Pending', 'Generated')),
  riskassessment VARCHAR(5000),
  repairactions VARCHAR(5000),
  inspectionschedule VARCHAR(5000),
  imageurl VARCHAR(300),
  crackmaskurl VARCHAR(300),
  overlayurl VARCHAR(300)
);
```

Make sure to Disable RLS.

Go to Storage and create 3 buckets named `original-images`, `crack-masks` and `overlay_images`

## Run application

### Engineering dashboard
Demo: https://www.youtube.com/watch?v=VG-mAhJgPKI
```
streamlit run app.py
```

### Admin dashboard
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

