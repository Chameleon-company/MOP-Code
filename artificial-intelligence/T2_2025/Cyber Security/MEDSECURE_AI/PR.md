# Pull Request – MedSecure AI Final Submission

## Overview
This PR finalises the MedSecure AI use case for the MOP AI & IoT Team – City of Melbourne.
The project delivers an AI-powered security API for detecting DDoS and Botnet attacks in smart public health infrastructure. It integrates trained deep learning models with a FastAPI microservice, making the system extensible, testable, and ready for deployment.

## Changes Implemented
- Created FastAPI microservice with modular architecture (config.py, inference_service.py, routers).
- Integrated trained models (ONNX / PyTorch) with auto-detection and optional vectorizer (joblib).
- Added endpoints:
  - /health/ready and /health/live for liveness/readiness checks
  - /v1/infer/batch for batch inference
  - /v1/ingest/syslog and /v1/ingest/agent for telemetry ingestion
- Implemented secure API key authentication middleware.
- Integrated Prometheus metrics via /metrics.
- Enabled ngrok tunneling for public testing in Colab.
- Wrote complete README (overview, setup, usage, known bugs, future work).

## Testing Done
- Verified model upload and auto-detection flow using Colab files.upload().
- Tested API health endpoints (/health/ready, /health/live).
- Ran inference on sample traffic records → correctly returned Normal, Botnet, or DDoS.
- Verified syslog and agent ingestion endpoints parse and log input successfully.
- Smoke-tested ngrok public URL with external curl requests.

## Known Issues
- Preprocessing pipeline is simplified; may not perfectly match training feature engineering.
- Colab runtime resets require re-upload of models.
- No large-scale load/stress testing has been conducted yet.
- Alerting/logging system is minimal (console logs only).
- Dependency on ngrok limits production readiness.

## Future Work
- Improve preprocessing to replicate full training pipeline feature engineering.
- Containerise project (Dockerfile) for deployment to cloud/on-premise infrastructure.
- Integrate advanced alerting and notifications.
- Conduct stress testing with benchmark datasets (CICIDS2018, CSIC 2010, TON_IoT).
- Expand to additional attack classes beyond DDoS and Botnet.
- Real-world integration with hospital IoT telemetry feeds.

## Contributors
- Arkhum Shahzad – Model training, API integration, FastAPI deployment, pipeline debugging
- Mi Vo – Dataset preparation (TON_IoT, NSL-KDD), preprocessing integration, documentation
