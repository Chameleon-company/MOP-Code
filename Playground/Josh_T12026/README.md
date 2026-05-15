# Smart Streetlight Fault Detection

An AI-powered streetlight monitoring system that uses Computer Vision and Large Language Models (LLMs) to detect faulty streetlights from nighttime images and generate maintenance reports.

## Features

- Upload nighttime streetlight images
- Detect streetlights using YOLO
- Classify lights as:
  - ON
  - DIM
  - OFF
- Generate AI maintenance reports using GPT
- Store report history using SQLite
- View uploaded images and previous reports

---

## Usage

1. Open the Streamlit application
2. Navigate to the Detection page
3. Upload one or more nighttime images
4. Click **Analyse image(s)**
5. View:
   - Detection results
   - Streetlight classifications
   - AI-generated maintenance reports

---

## Project Structure

```text
project/
│
├── backend_api/
│   ├── api.py
│   └── models/
│       ├── cv_model.py
│       └── best.pt
│
├── LLM/
│   └── llm_reporting.py
│
├── User_Interface/
│   └── ui.py
│
├── uploads/
├── reports.db
├── .env
└── README.md
```

---

## Running the Application

### Run Streamlit

```bash
streamlit run User_Interface/ui.py
```

### Run FastAPI

```bash
uvicorn backend_api.api:app --reload
```