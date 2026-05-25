# 🏗️ Crack Detection & Analysis System

## 📖 Overview
This project analyses cracks in infrastructure images (roads, bridges, surfaces) and generates both **visual outputs** and **AI-based insights**.

It combines Computer Vision and AI to:
- Detect crack regions
- Extract key features (length, width, severity)
- Generate processed images
- Provide intelligent explanations via chatbot

---

## 🚀 Features

- Detects cracks from uploaded images
- Extracts:
  - **Crack Length**
  - **Crack Width**
  - **Severity Level**
- Generates visual outputs:
  - **Binary Mask**
  - **Crack Overlay**
- AI-powered chatbot for:
  - Risk analysis
  - Recommendations
  - Engineering explanations
- Supports structured dataset storage

---

## 🛠️ Requirements

Install required libraries:

```bash
pip install fastapi uvicorn requests numpy opencv-python pillow plotly openai