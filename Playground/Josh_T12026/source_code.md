========================================
FILE: backend_api/api.py
========================================

import sqlite3
import json
import os 
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from typing import List
from fastapi.staticfiles import StaticFiles
from backend_api.models.cv_model import analyse_image
from LLM.llm_reporting import llm_reporting
import base64

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
            results TEXT
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
            "analysis": analysis, 
            "uploaded_img": base64.b64encode(contents).decode("utf-8")
        })

        saved_files.append(file.filename)
    
    return {        
        "report_id": report_id, 
        "results": results
    }
    
#REPORT ENDPOINT FOR LLM 
@app.post("/report")
async def generate_report(data: dict): 
    report_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_results = []

    for item in data["results"]:

        analysis = item["analysis"]

        llm_result = await llm_reporting({
            "analysis": analysis,
            "uploaded_img": item.get("uploaded_img")
        })

        final_results.append({
            "image": item["image"],
            "analysis": analysis,
            "report": llm_result["output"]
        })
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reports (report_id, timestamp, results)
        VALUES (?, ?, ?)
    """, (
        report_id,
        datetime.now().isoformat(),
        json.dumps(final_results)
    ))

    conn.commit()
    conn.close()
        
    return {
        "report_id": report_id,
        "results": final_results
    }
    
@app.get("/reports")
def get_reports(): 
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    #query all reports 
    cursor.execute("""
        SELECT report_id, timestamp, results
        FROM reports
        ORDER BY id DESC
    """)
    
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



========================================
FILE: backend_api/models/cv_model.py
========================================

import os
from ultralytics import YOLO
import cv2
import numpy as np
import base64

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# load YOLO model ONCE
model = YOLO(MODEL_PATH)

# -----------------------------
# preprocessing functions
# -----------------------------

def brighten_image(image, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    l_clahe = clahe.apply(l)

    merged = cv2.merge((l_clahe, a, b))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def brighten_then_clahe(image):
    bright = brighten_image(image)
    return apply_clahe(bright)

# temporary substitute if missing
def double_check_preprocess(image):
    return brighten_then_clahe(image)

# -----------------------------
# classification
# -----------------------------

def classify_streetlight_state(
    crop,
    off_threshold=80,
    dim_threshold=160
):

    if crop is None or crop.size == 0:
        return "unknown", 0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    pixels = gray.flatten()
    pixels = np.sort(pixels)

    top_pixels = pixels[int(0.9 * len(pixels)):]

    mean_brightness = float(np.mean(top_pixels))

    if mean_brightness < off_threshold:
        state = "off"

    elif mean_brightness < dim_threshold:
        state = "dim"

    else:
        state = "on"

    return state, mean_brightness


# -----------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------

def analyse_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return {
            "error": "Could not load image"
        }

    processed_img = double_check_preprocess(image)

    results = model(
        processed_img,
        conf=0.10,
        verbose=False
    )

    boxes = results[0].boxes
    
    if boxes is None or len(boxes) == 0: 
        return {
            "streetlight_count": 0, 
            "on": 0,
            "dim": 0,
            "off": 0,
            "details": []
        }

    on_count = 0 
    dim_count = 0 
    off_count = 0 

    details = []

    for box in boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        crop = image[y1:y2, x1:x2]

        state, brightness = classify_streetlight_state(crop)

        if state == "on":
            on_count += 1

        elif state == "dim":
            dim_count += 1

        elif state == "off":
            off_count += 1

        details.append({
            "bbox": [x1, y1, x2, y2],
            "state": state,
            "brightness": brightness
        })

    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(
            img_file.read()
        ).decode("utf-8")

    return {
        "uploaded_img": encoded_image,
        "streetlight_count": len(boxes),
        "on": on_count,
        "dim": dim_count,
        "off": off_count,
        "details": details
    }


========================================
FILE: LLM/llm_reporting.py
========================================

import os
import json

from openai import AsyncOpenAI

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def gen_prompt(data):

    analysis = data["analysis"]
    
    prompt = f"""
    You are an AI-powered streetlight monitoring assistant.

    An uploaded streetlight image is provided together with ML detection results.

    ML Detection Results:
    - Total Streetlights: {analysis["streetlight_count"]}
    - ON Lights: {analysis["on"]}
    - DIM Lights: {analysis["dim"]}
    - OFF Lights: {analysis["off"]}
    - Detection Details: {analysis["details"]}

    Instructions:
    - Analyze the uploaded image together with the ML output.
    - Describe the overall streetlight condition naturally.
    - Mention operational, dim, and faulty streetlights.
    - Highlight maintenance concerns if necessary.
    - Keep the response concise and professional.
    - Do not mention Base64 data.

    Return ONLY valid JSON:

    {{
        "output": "your report here"
    }}
    """

    return prompt


async def gpt(prompt, base64_image):

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    completion = await client.chat.completions.create(
        model="gpt-4.1-mini",

        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],

        max_tokens=800,
        temperature=0
    )

    response = completion.choices[0].message.content

    return json.loads(response)


async def llm_reporting(data):

    base64_image = data["uploaded_img"]

    prompt = await gen_prompt(data)

    response = await gpt(prompt, base64_image)

    return response


========================================
FILE: User_Interface/ui.py
========================================

import streamlit as st 
import requests
import base64

#PAGECONFIG
st.set_page_config(
    page_title = "Smart Streetlight Fault Detection",
    layout = "centered",
    initial_sidebar_state = "expanded"
)

#send images to FastAPI
def backend_detect(files):
    try: 
        url = "http://localhost:8000/detect"
        response = requests.post(url, files = files)
        return response.json()
    except: 
        return {"error": "Backend unavailable"}

#send detection results to FastAPI
def backend_report(detection_data):
    try: 
        url = "http://localhost:8000/report"
        response = requests.post(url, json = detection_data)
        return response.json()
    except: 
        return {"error": "Backend unavailable"}

def backend_get_reports():
    try:
        response = requests.get("http://localhost:8000/reports")
        return response.json()
    except:
        return {"reports": []}

#SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Detection", "Reports", "Report History", "About"])

#HOMEPAGE 
if page == "Homepage": 
    st.title(":blue[Smart Streetlight Fault Detection] :bulb:", text_alignment = "center")
    
    st.markdown("""
                This project utilises a vision-based system to analyse provided nighttime images and detects faulty streetlights, specifically: 
                - Streetlights that are either 
                    - **not functioning** 
                    - **flickering** 
                    - **producing a weak illumination**
                
                Following this, a maintenence alert will be generated. 
                
                Please proceed to the detection page to get started. """)

#DETECTIONPAGE
elif page == "Detection": 
    st.title("Streetlight Analysis")
    
    #COLUMNS
    col1, col2 = st.columns([1, 2])
    
    with col1: 
    #file uploader
        uploaded_files = st.file_uploader("Upload image(s)", accept_multiple_files = True, type = ["jpg", "png"])

    with col2: 
    #loop through uploaded files, display preview, and add a button 
        if uploaded_files: 
            st.header("Preview uploaded images")
            for file in uploaded_files: 
                st.image(file, width="stretch")
                st.markdown("---")
            
            #button for analysis 
            if st.button("Analyse image(s)", type = "primary"):
                with st.spinner("Analysing images..."):
                    #prep files 
                    file_data = [
                        ("files", (file.name, file.getvalue(), file.type))
                        for file in uploaded_files
                    ]
                    
                    #send to backend
                    result = backend_detect(file_data)
                    if isinstance(result, dict) and "results" in result:
                        st.session_state["analysis_result"] = result
                        st.success("Analysis complete!")
                    else:
                        st.error("Backend returned invalid response")
                        st.json(result)
        else:
            st.info("No uploaded images.")

    #DIVIDE PAGE 
    st.markdown("---")

    #analysis results section 
    st.header("Analysis Results")
    if "analysis_result" in st.session_state:
        result_data = st.session_state["analysis_result"]

        if isinstance(result_data, dict) and "results" in result_data:
            for result in result_data["results"]:

                analysis = result["analysis"]
                st.subheader(result["image"])

                #if an image exists 
                if "uploaded_img" in analysis:
                    image_bytes = base64.b64decode(
                        analysis["uploaded_img"]
                    )
                    st.image(image_bytes)

                st.write(f"Streetlights: {analysis['streetlight_count']}")
                st.write(f"On: {analysis['on']}")
                st.write(f"Dim: {analysis['dim']}")
                st.write(f"Off: {analysis['off']}")

                st.json(analysis["details"])

                st.markdown("---")
    else:
        st.info("No analysis results")

#REPORTSPAGE
elif page == "Reports": 
    st.title("Maintenance Report")
    
    if st.button("Generate report", type = "primary"): 
        if "analysis_result" not in st.session_state: 
            st.warning("No results available. Please run detection analysis first")
        else: 
            with st.spinner("Generating report..."):
                detection_data = st.session_state['analysis_result']
                
                #send to backend 
                report = backend_report(detection_data)
                
            if isinstance(report, dict) and "results" in report:
                st.success("Report generated!")

                for item in report["results"]: 
                    st.subheader(item["image"])
                    st.write(item["report"])
                    st.markdown("---")
            else:
                st.error("Report failed or invalid response from backend")
                st.json(report)
    
#REPORTHISTORY
elif page == "Report History": 
    st.title("Report History")
    data = backend_get_reports() 
    reports = data.get("reports", [])
    
    if not reports: 
        st.warning("No reports found. ")
    else: 
        for report in reports:
            st.subheader(f"Report: {report['report_id']}")

            for item in report["results"]:

                image_url = f"http://localhost:8000/uploads/{report['report_id']}/{item['image']}"

                st.image(image_url)

                st.json(item["analysis"])

                if "report" in item:
                    st.success(item["report"])
                else:
                    st.warning("No LLM report available")

#ABOUTPAGE
elif page == "About": 
    st.title(":blue[About Us]", text_alignment = "center")
    
    st.markdown("""
            Our project team is composed of the following members: 
            
            **Syed Hamiz Hassan** 
            - Computer Vision Modelling 
            
                
            **Savith Mundukotuwa** 
            - Data Preparation 
            
            
            **Josh Wong** 
            - User Interface and System Integration
            
            
            **Luke Kankannamge Don** 
            - LLM Reporting System 
            
            
            **Rahul Sheoran** 
            - Model Analysis
             
            """)


========================================
FILE: .env.example
========================================

OPENAI_API_KEY=your_api_key_here