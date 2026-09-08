from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException


app = FastAPI(title="Resume Job Matcher Data API")

BASE_DIR = Path(__file__).resolve().parent
JOBS_FILE = BASE_DIR / "data" / "jobs.csv"
RESUME_FILE = BASE_DIR / "data" / "resume.txt"


@app.get("/")
def root():
    return {
        "message": "Resume Job Matcher Data API",
        "endpoints": ["/health", "/resume/default", "/jobs"],
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/resume/default")
def get_default_resume():
    if not RESUME_FILE.exists():
        raise HTTPException(status_code=404, detail="Resume file not found")

    return {"resume_text": RESUME_FILE.read_text(encoding="utf-8")}


@app.get("/jobs")
def get_jobs():
    if not JOBS_FILE.exists():
        raise HTTPException(status_code=404, detail="Jobs file not found")

    jobs_df = pd.read_csv(JOBS_FILE)
    required_columns = {"title", "company", "description"}
    missing_columns = required_columns - set(jobs_df.columns)

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"jobs.csv is missing required columns: {sorted(missing_columns)}",
        )

    return {"jobs": jobs_df.to_dict(orient="records")}
