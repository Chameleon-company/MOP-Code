from flask import Flask, request, jsonify
import random
from dotenv import load_dotenv
import os
from supabase import create_client, Client
from flask_cors import CORS
from pathlib import Path
from crackAnalyser import generateMetricReport
from PIL import Image
import io


app = Flask(__name__)
CORS(app, supports_credentials=True)

#Supdabase database connection 
result = load_dotenv(Path(__file__).parent / '.env.local', verbose=True)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY")

print("Supabase URL:", SUPABASE_URL)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)








def RandomSentenceGen(wordCount):
    words = ["hello", "dictionary", "groan", "bounce", "pull", "surface", "salvation", "get", "acceptable", "retailer", "murder", "magnitude", "award", "score", "collection", "hard", "formation", "marine", "notorious", "lily", "gate", "earthquake"]
    sentence = ""
    i = 0
    while i < wordCount:
        word = random.choice(words)
        word += " "
        sentence += word
        i += 1
        
    return sentence
    

class CrackReport():
    def __init__(self, crackAreaRatio, estimatedCrackLength, numCracks, severity, modelConfidence, riskManagement, recommendedRepair, nextAction):
        self.crackAreaRatio = crackAreaRatio
        self.estimatedCrackLength = estimatedCrackLength
        self.numCracks = numCracks
        self.severity = severity
        self.modelConfidence = modelConfidence
        self.riskManagement = riskManagement
        self.recommendedRepair = recommendedRepair
        self.nextAction = nextAction
        
    def __str__(self):
        return f"Crack Area = {self.crackAreaRatio}, Estimated crack length = {self.estimatedCrackLength}, Number of cracks = {self.numCracks}, Severity = {self.severity}, Model Confidence = {self.modelConfidence}, Risk Management = {self.riskManagement}, Recommended Repair = {self.recommendedRepair}, Next Action = {self.nextAction}"


def genRandomReport():
    #severitys = ["low", "medium", "high", "critical"]
    crackReport = CrackReport(
        crackAreaRatio = random.randrange(1, 100) / 100,
        estimatedCrackLength = random.randrange(1, 1000),
        numCracks = random.randrange(1, 10),
        severity = random.randrange(1,4),
        modelConfidence = random.randrange(1, 100) / 100,
        riskManagement = RandomSentenceGen(30),
        recommendedRepair = RandomSentenceGen(30),
        nextAction = RandomSentenceGen(30)
    )
    
    return crackReport
    
    
def uploadReport(report):
    print(f"Report = {report}")
    try:
        data = supabase.table('crack_reports').insert({
            'imageid': report["image_id"],
            'severity': report["severity"],
            'numcracks': report["num_crack_regions"],
            'crackarearatio': report["largest_crack_area_ratio"],
            'estimatedcracklength': report["largest_crack_est_length"],
            'riskmanagement': report["riskManagement"],
            'recommendedrepair': report["recommendedRepair"],
            'nextaction': report["nextAction"]            
        }).execute()
    except Exception as e:
        raise

    
    
@app.get("/api/ping")
def ping():
    return jsonify("Hello from the Road Crack Detection Project!"), 200

@app.get("/api/genRandomReport")
def generateRandomReport():
    try:
        report = genRandomReport()
        try:
            uploadReport(report)
            
        except Exception as e:
            print(str(e))
            return jsonify(f"Error uploading to database: {e}"), 500
        
        
    except Exception as e:
        return jsonify(f"Error: {e}"), 500
    
    return jsonify(str(report)), 200


@app.route("/api/getData")
def get_data():
    data = supabase.table("crack_reports").select("*").execute()
    if data.data is None:
        return jsonify({"error": "Couldnt load crack reports"}), 500
    
    print(str(data))
    return jsonify({"Data": data.data})




@app.post("/api/uploadImage")
def uploadImage():
    file = request.files["file"]
    filename = file.filename 
    imageBytes = file.read()
    image = Image.open(io.BytesIO(imageBytes))
    
    try:
        report = generateMetricReport(image, filename)
    except Exception as e:
        return jsonify(f"Error generating report: {e}"), 500

    if report["crack_detected"] == False:
        return jsonify("No crack detected in image"), 200
    
    elif report["crack_detected"] == True:
        try:
            del report["crack_detected"]
            report["riskManagement"] = "PLACEHOLDER"
            report["recommendedRepair"] = "PLACEHOLDER"
            report["nextAction"] = "PLACEHOLDER"
            
            uploadReport(report)
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