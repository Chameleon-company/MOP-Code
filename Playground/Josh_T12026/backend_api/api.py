import sqlite3
import json
import os 
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from typing import List
from fastapi.staticfiles import StaticFiles
from models.cv_model import analyse_image

#start API app
app = FastAPI()

upload_directory = "uploads"
os.makedirs(upload_directory, exist_ok = True)

app.mount("/uploads", StaticFiles(directory = "uploads"), name = "uploads")

db_path = "reports.db"

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT,
            timestamp TEXT,
            images TEXT
        )
    """)

    conn.commit()
    conn.close()
    
init_db()

@app.get("/")
def root():
    return {"backend": "working"}

#DETECTION ENDPOINT FOR CV
@app.post("/detect")
async def detect_lights(files: List[UploadFile] = File(...)):
    #create unique report folders for each individual report formatted y/m/d (year month day) and h/m/s (hours minutes seconds)
    report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(upload_directory, report_id)
    os.makedirs(report_path, exist_ok = True)
    
    #for storing in the database and displaying results 
    saved_files = []
    results = []
    
    for file in files: 
        contents = await file.read() 

        file_path = os.path.join(report_path, file.filename)

        with open(file_path, "wb") as f: 
            f.write(contents)

        #run CV model analysis
        analysis = analyse_image(file_path)

        results.append({
            "image": file.filename,
            "analysis": analysis
        })

        saved_files.append(file.filename)
    
    #save to SQLite db file
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO reports (report_id, timestamp, images)
        VALUES (?, ?, ?)
    """, (
        report_id,
        datetime.now().isoformat(),
        json.dumps(results)
    ))

    conn.commit()
    conn.close()
    
    return {        
        "report_id": report_id, 
        "results": results
    }

@app.get("/reports")
def get_reports(): 
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    #query all reports 
    cursor.execute("SELECT report_id, timestamp, images FROM reports ORDER BY id DESC")
    rows = cursor.fetchall()
    
    conn.close()
    
    reports = []
    for row in rows:
        reports.append({ 
            "report_id": row[0],
            "timestamp": row[1],
            "results": json.loads(row[2])
        })
    
    return {"reports": reports}

#REPORT ENDPOINT FOR LLM 
@app.post("/report")
async def generate_report(): 
    return { 
        #placeholder results
        "report": "placeholder report"
    }