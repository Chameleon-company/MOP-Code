import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from supabase import create_client, Client
from flask_cors import CORS
from pathlib import Path
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
import tempfile
import cv2
import numpy as np
# ── TO-DO ──────────────    
# - UploadImage should use generateAiReports function
# ── Local imports ────────────────────
from Metric_Generator.crackAnalyser import generateMetricReport
from LLM_pipeline.llm import report_generation
from reportGenerationHelpers import fetchSingleRow, uploadReport, changeRowStatus, convertReport, generateAiReport, updateRowWithAiReport, noCrackReport
from crack_detection.pipeline import run_pipeline

app = Flask(__name__)
CORS(app, supports_credentials=True)

# ── Supabase Setup ────────────────────
result = load_dotenv(Path(__file__).parent / '.env.local', verbose=True)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY")
print("Supabase URL:", SUPABASE_URL)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Threading setup (Used by background AI reports) ────────────────────
executor = ThreadPoolExecutor(max_workers=4)

# ── Ping Endpoint ──────────────    
@app.get("/api/ping")
def ping():
    return jsonify({"message": "Hello from the Road Crack Detection Project!"}), 200

# ── Get all rows ──────────────    
@app.route("/api/getData")
def get_data():
    try:
        data = supabase.table("crack_reports").select("*").execute()
        if data.data is None:
            return jsonify({"error": "Couldnt load crack reports. No Data returned"}), 404
        return jsonify({"Data": data.data})
    except ConnectionError as e:
        return jsonify({"error": f"Couldnt connect to supabase: {e}"}), 500
    except TimeoutError as e:
        return jsonify({"error": f"Connection timed out: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Undefined Error: {e}"}), 500
    

# ── Get all rows without an AI report ──────────────    
@app.route("/api/getNoReportData")
def get_no_report_data():
    try:
        data = supabase.table("crack_reports").select("*").eq("reportstatus", "None").execute()
        if data.data is None:
            return jsonify({"error": "Couldnt load crack reports. No Data returned"}), 500
        return jsonify({"Data": data.data})
    except ConnectionError as e:
        return jsonify({"error": f"Couldnt connect to supabase: {e}"}), 500
    except TimeoutError as e:
        return jsonify({"error": f"Connection timed out: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Undefined Error: {e}"}), 500
    
    
# ── Generate AI report in the background ──────────────    
@app.patch("/api/generateAiReport")
def generateAiReports():
    try:
        data = request.get_json()
        idArray = data.get("IDs")
        print(idArray)
        reports = []
        rows = []
        for id in idArray:
            row = fetchSingleRow(supabase, id)
            if row == None:
                return jsonify({"error": f"No row with id {id} could be found"}), 500
            else:
                rows.append(row)
    except Exception as e:
        print(e)
        return jsonify({"error": f"Error: {e}"}), 500

    def executeReportgeneration(report):
        try:
            changeRowStatus(supabase, report["id"], "Pending")
        except Exception as e:
            print({"error": f"Error updating status to pending: {e}"})
        try:
            convertedReport = convertReport(report)
            newReport = generateAiReport(convertedReport)
            updateRowWithAiReport(supabase, newReport)
        except Exception as e:
            try:
                changeRowStatus(supabase, report["id"], "None")
                print({"error": f"Error generating AI report: {e}"})
            except:
                print({"error": f"CRITICAL ERROR, Could change report status back to None: {e}"})

    for row in rows:
        executor.submit(executeReportgeneration, row)
    return jsonify({"message": "Report Generation started"}), 202

# ── Take image, convert to binary mask, generate metric and if selected generate AI report ──────────────    
@app.post("/api/uploadImage")
def uploadImage():
    flag = request.args.get("flag", "false").lower() == "true"
    file = request.files["file"]
    filename = file.filename
    imageBytes = file.read()
    image = Image.open(io.BytesIO(imageBytes))

    if image is not None:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        image.save(temp_path)
        result = run_pipeline(temp_path)
    else:
        return jsonify({"error": f"Image was None"}), 400

    try:
        mask_url = result["mask_url"]
        img_url = result["original_url"]
        overlay_url = result["overlay_url"]                  # NEW
        response = requests.get(mask_url)
        mask_png = response.content
        mask = Image.open(io.BytesIO(mask_png))
    except Exception as e:
        return jsonify({"error": f"Error retrieving crack mask from supabase: {e}"}), 500

    try:
        print("Generating Crack metrics")
        report = generateMetricReport(mask, filename)
        report["image_url"] = img_url
        report["mask_url"] = mask_url
        report["overlay_url"] = overlay_url                  # NEW
        print(f"Mask Url = {mask_url}")
        print(f"Original Url = {img_url}")
        print(f"Overlay Url = {overlay_url}")                # NEW
        print(f"report = {report}")
    except Exception as e:
        return jsonify({"error": f"Error generating report: {e}"}), 500

    if report["crack_detected"] == False:
        report = noCrackReport(report)
        uploadReport(supabase, report)
        return jsonify("No crack detected in image"), 200

    elif report["crack_detected"] == True:
        try:
            del report["crack_detected"]
            if flag:
                try:
                    llm_result = report_generation(report)
                    sections = llm_result['sections']
                    report["report_status"] = "Generated"
                    report["risk_assessment"] = sections['risk_assessment']['reasoning']
                    report["repair_actions"] = sections['repair_actions']['reasoning']
                    report["inspection_schedule"] = sections['inspection_schedule']['reasoning']
                except Exception as e:
                    report["report_status"] = "None"
                    report["risk_assessment"] = ""
                    report["repair_actions"] = ""
                    report["inspection_schedule"] = ""
                    print(f"Exception: {e}")
            else:
                report["report_status"] = "None"
                report["risk_assessment"] = ""
                report["repair_actions"] = ""
                report["inspection_schedule"] = ""
            uploadReport(supabase, report)
            return jsonify(report), 201
        except AttributeError as e:
            return jsonify({"error": f"Missing / invalid field on report object: {e}"}), 500
        except ConnectionError as e:
            return jsonify({"error": f"Error connection to Supabase database {e}"}), 500
        except TimeoutError as e:
            return jsonify({"error": f"Connection timed out {e}"}), 500
        except Exception as e:
            return jsonify({"error": f"Unknown error occurred uploading to database: {e}"}), 500
    else:
        return jsonify({"error": f"No crack_detected field found"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)