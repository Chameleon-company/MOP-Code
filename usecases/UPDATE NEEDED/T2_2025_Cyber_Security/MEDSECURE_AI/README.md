# MedSecure AI – AI-Powered Security for Smart Public Health Infrastructure

## Project Overview
MedSecure AI is part of the Melbourne Open Data Playground (MOP) AI & IoT project stream. The goal of this use case is to design and implement an AI-powered intrusion detection and response system tailored for smart healthcare environments.

Problem:
Modern hospitals and healthcare IoT ecosystems face increasing risks from DDoS, Botnet, and other network-based cyberattacks. These threats can disrupt critical medical services, compromise patient data, and reduce system reliability.

Approach:
- Developed a modular FastAPI-based service that exposes a machine learning inference pipeline via secure REST APIs.
- Integrated trained deep learning models (LSTM, BiLSTM, CNN-LSTM variants) for classification of traffic as Normal, Botnet, or DDoS.
- Enabled preprocessing pipeline integration with optional vectorizers (joblib).
- Added monitoring and observability with Prometheus and liveness/readiness endpoints.
- Exposed endpoints for inference, syslog ingestion, and agent event handling.
- Deployed and tested in Google Colab with ngrok for external access.

Outcome:
- A working proof-of-concept security API that accepts real traffic data, runs through trained ML pipelines, and outputs detection results.
- Demonstrated integration with datasets such as CICIDS2018, CSIC 2010, and TON_IoT for training and evaluation.
- Provided a scalable, extensible design that can be containerised and deployed into larger smart healthcare IoT systems.

## Environment Setup and Installation

### Requirements
- Python 3.9+
- Dependencies:
  - fastapi, uvicorn
  - onnxruntime, torch
  - joblib, scikit-learn
  - loguru, orjson
  - pyngrok (for public testing in Colab)
  - starlette-exporter (for Prometheus metrics)

Install in Colab or local environment:
```bash
pip install fastapi uvicorn loguru orjson python-multipart starlette-exporter requests
pip install onnxruntime torch joblib scikit-learn pyngrok
```

## How to Run the Code

1. Upload model and vectorizer files (if applicable)
```python
from google.colab import files
uploaded = files.upload()
```

2. Move files and auto-detect paths
```python
import os, shutil
os.makedirs("/content/models", exist_ok=True)
for name in uploaded.keys():
    shutil.move(name, f"/content/models/{name}")
```

3. Launch FastAPI app
```python
get_ipython().system_raw("cd /content/medsecure_api && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &")
```

4. Test locally
```python
import requests, time, os
time.sleep(2)
base = "http://127.0.0.1:8000"
print(requests.get(f"{base}/health/ready").json())
```

5. Optional – Enable ngrok tunnel
```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
public_url = ngrok.connect(8000)
print("Public URL:", public_url)
```

6. Send inference request
```python
payload = {
    "records":[
        {"timestamp":"2025-09-24T08:00:00Z",
         "src_ip":"10.0.0.5","dst_ip":"10.0.0.7",
         "method":"POST","url":"/login","payload":"bot ping payload"}
    ]
}
r = requests.post(f"{base}/v1/infer/batch",
                  headers={"X-API-Key":"medsecure_dev_key"},
                  json=payload)
print(r.json())
```

## Known Bugs or Limitations
- Preprocessing pipeline is generic; may not perfectly replicate training feature engineering.
- Colab runtime resets clear the running server and require re-upload of models.
- API performance not yet stress-tested under high traffic.
- Logging and alerting integrations are minimal (basic console logs only).
- Dependency on ngrok for public testing, not suitable for production.

## Future Improvements and Plans
- Extend preprocessing to fully align with training feature engineering steps.
- Containerise (Dockerfile) for deployment in cloud or on-prem healthcare infrastructure.
- Add alerting and notification system for real-time attack response.
- Scale testing using benchmark datasets to validate detection accuracy under load.
- Expand model to detect additional classes of attacks beyond Botnet and DDoS.
- Integrate with hospital IoT telemetry feeds for real-world trials.

## Contributors
- Arkhum Shahzad – Model training, API integration, FastAPI deployment in Colab, pipeline debugging
- Mi Vo – Dataset preparation (TON_IoT, NSL-KDD), integration with preprocessing pipeline, project documentation
