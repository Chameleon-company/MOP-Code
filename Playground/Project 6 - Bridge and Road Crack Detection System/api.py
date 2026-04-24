
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
# - Image to Binary Mask conversion needs to be added to image upload
# - Image upload also needs to upload image to seperate supabase database




# ── Local imports ────────────────────
from Metric_Generator.crackAnalyser import generateMetricReport
from LLM_pipeline.llm import report_generation
from reportGenerationHelpers import fetchSingleRow, uploadReport, changeRowStatus, convertReport, generateAiReport, updateRowWithAiReport
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
    return jsonify("Hello from the Road Crack Detection Project!"), 200



# ── Get all rows ──────────────    
@app.route("/api/getData")
def get_data():
    try:
        data = supabase.table("crack_reports").select("*").execute()
        if data.data is None:
            return jsonify({"error": "Couldnt load crack reports"}), 500
    
        return jsonify({"Data": data.data})
    except ConnectionError as e:
        return jsonify(f"Error connection to Supabase database {e}"), 500
    except TimeoutError as e:
        return jsonify(f"Connection timed out {e}"), 500
    except Exception as e:
        return jsonify({"error": "Couldnt load crack reports"}), 500

# ── Get all rows without an AI report ──────────────    
@app.route("/api/getNoReportData")
def get_no_report_data():
    try:
        data = supabase.table("crack_reports").select("*").eq("reportstatus", "None").execute()
        if data.data is None:
            return jsonify({"error": "Couldnt load crack reports"}), 500
        
        return jsonify({"Data": data.data})
    except ConnectionError as e:
        return jsonify(f"Error connection to Supabase database {e}"), 500
    except TimeoutError as e:
        return jsonify(f"Connection timed out {e}"), 500
    except Exception as e:
        return jsonify({"error": "Couldnt load crack reports"}), 500 
    
# ── Generate AI report in the background ──────────────    
@app.patch("/api/generateAiReport")
def generateAiReports():
    try:
        #Get ID's
        data = request.get_json()
        idArray = data.get("IDs")
        print(idArray)
        
        reports = []
        rows = []
        
        #Check if id's are in the database
        for id in idArray:
            row = fetchSingleRow(supabase, id)
            if row == None:
                return jsonify(f"No row with id {id} could be found: {e}"), 500
            else:
                rows.append(row)
                
    except Exception as e:
        print(e)
        return jsonify(f"Error: {e}"), 500
    
    
    def executeReportgeneration(report):
        #Change status to pending
        try:
            changeRowStatus(supabase, report["id"], "Pending")
        except Exception as e:
            print(f"Error updating status to pending: {e}")
        
        #Convert report to correct format, generate AI report and upload it
        try:
            convertedReport = convertReport(report)
            newReport = generateAiReport(convertedReport)
            updateRowWithAiReport(supabase, newReport)
            
        #If anything goes wrong status must be reverted back to None
        except Exception as e:
            try:
                changeRowStatus(supabase, report["id"], "None")
                print(f"Error generating AI report: {e}")
            except:
                print(f"CRITICAL ERROR, Could change report status back to None: {e}")
    
    #For each row that corresponds with the uploaded ID's run reeport generation (multi threaded)
    for row in rows:
        executor.submit(executeReportgeneration, row)       

    return jsonify({"message": "Report Generation started"}), 202
 

# ── Take image, convert to binary mask, generate metric and if selected generate AI report ──────────────    
@app.post("/api/uploadImage")
def uploadImage():
    #Get flag, if flag is true AI report generation should also be executed
    flag = request.args.get("flag", "false").lower() == "true"
    file = request.files["file"]
    filename = file.filename 
    
    imageBytes = file.read()
    image = Image.open(io.BytesIO(imageBytes))
    
    #Image to crack mask conversion.
    if image is not None:   
        temp_dir = tempfile.gettempdir() 
        temp_path = os.path.join(temp_dir, filename)
        image.save(temp_path)
        result = run_pipeline(temp_path)      
        
    else:
        return jsonify(f"Image was None: {e}"), 400
    
    try:
        mask_url = result["mask_url"]
        response = requests.get(mask_url)
        mask_png = response.content
        mask = Image.open(io.BytesIO(mask_png))
    except Exception as e:
        return jsonify(f"Error retrieving crack mask from supabase: {e}"), 500
    
    #Generate crack metrics
    try:
        print("Generating Crack metrics")
        report = generateMetricReport(mask, filename)
    except Exception as e:
        return jsonify(f"Error generating report: {e}"), 500

    if report["crack_detected"] == False:
        return jsonify("No crack detected in image"), 200
    
    #upload, if flag is true start AI report generation
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
            return jsonify(f"Missing / invalid field on report object: {e}"), 500
        except ConnectionError as e:
            return jsonify(f"Error connection to Supabase database {e}"), 500
        except TimeoutError as e:
            return jsonify(f"Connection timed out {e}"), 500
        except Exception as e:
            return jsonify(f"Unknown error occurred uploading to database: {e}"), 500
    else:
        return jsonify(f"No crack_detected field found"), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)