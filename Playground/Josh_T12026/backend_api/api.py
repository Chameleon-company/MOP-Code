from fastapi import FastAPI, UploadFile, File
from typing import List

app = FastAPI()

@app.get("/")
def root():
    return {"backend": "working"}

#DETECTION ENDPOINT FOR CV
@app.post("/detect")
async def detect_lights(files: List[UploadFile] = File(...)):
    return {        
        #placeholder results
        "total_lights": 3,
        "faulty_lights": 1, 
    }
    
#REPORT ENDPOINT FOR LLM 
@app.post("/report")
async def generate_report(): 
    return { 
            #placeholder results
        "placeholder report"
    }