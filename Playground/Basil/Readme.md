# **Resume Job Matcher**

An NLP-based Resume-to-Job Matching Assistant that evaluates how well a candidate’s resume aligns with job descriptions using a combination of keyword matching, semantic similarity, skill analysis, and API-based data retrieval.

---

# **Overview**

This project started as a notebook-based prototype and evolved into a more complete decision-support system that combines multiple NLP techniques to generate a more balanced and interpretable **Job Fit Score (%)**.

The system helps users:

- Identify strong-fit roles
- Understand missing skills
- Compare multiple job opportunities
- Avoid applying to poorly matched positions

The project was later upgraded to include:

- A Streamlit-based interactive UI
- API-based data access using FastAPI
- Modular backend architecture
- Hybrid NLP scoring pipeline

---

# **Key Features**

## **1. Multi-Method Matching System**

The system combines multiple NLP approaches:

### **TF-IDF + Cosine Similarity**

Captures keyword-level similarity between the resume and job description.

Used as the baseline retrieval model.

---

### **Semantic Similarity (Sentence Embeddings)**

Uses `sentence-transformers` to capture contextual meaning and semantic relationships between skills, roles, and descriptions.

This improves matching beyond exact keyword overlap.

---

### **Skill Extraction & Gap Analysis**

The system extracts technical skills from both the resume and job descriptions using a curated skill list.

It identifies:

- Matched skills
- Missing skills
- Skill overlap score

This improves explainability and interpretability.

---

## **2. Hybrid Scoring Model**

The final **Job Fit Score (%)** combines:

- 50% Semantic similarity
- 30% Skill overlap
- 20% TF-IDF similarity

This balances:

- contextual understanding
- explicit skill matching
- keyword precision

---

## **3. Interactive Streamlit UI**

The project includes a lightweight Streamlit web application where users can:

- Upload resumes (`.txt` or `.pdf`)
- Paste resume text
- Paste job descriptions
- Generate:

  - Job Fit Score (%)
  - TF-IDF similarity
  - Semantic similarity
  - Skill overlap score
  - Matched skills
  - Missing skills

The UI transforms the notebook prototype into a practical demo-ready application.

---

## **4. API-Based Data Access**

The system was upgraded from direct local file loading to an API-based architecture using **FastAPI**.

Instead of loading data directly inside the notebook, the notebook and frontend can retrieve resume and job data through REST API endpoints.

Available endpoints include:

- `/resume/default`
- `/jobs`

This improves:

- modularity
- reproducibility
- maintainability
- real-world system design

---

# **Project Structure**

```text
resume_job_matcher/
│
├── data/
│   ├── jobs.csv
│   └── resume.txt
│
├── notebooks/
│   ├── resume_job_matching.ipynb
│   ├── resume_job_matching.json
│   └── resume_job_matching.html
│
├── outputs/
│   ├── final_prototype_results.csv
│   ├── ranked_jobs.csv
│   ├── prototype_results_chart.png
│   └── tfidf_vs_semantic_comparison.csv
│
├── api_server.py
├── app.py
├── requirements.txt
└── README.md
