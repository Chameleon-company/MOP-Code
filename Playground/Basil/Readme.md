
# **Resume Job Matcher**

An NLP-based Resume-to-Job Matching Assistant that evaluates how well a candidate’s resume aligns with a job description using a combination of keyword matching, semantic similarity, and skill analysis.

---

## **Overview**

This project started as a notebook-based prototype and was extended into a full system that combines multiple matching techniques to produce a more accurate and interpretable **job-fit score**.

The system helps users:

* Identify strong-fit roles
* Understand skill gaps
* Avoid applying to poorly matched jobs

---

## **Key Features**

### **1. Multi-Method Matching**

The system uses three complementary approaches:

* **TF-IDF + Cosine Similarity**
  Captures keyword-level relevance between resume and job description.

* **Semantic Similarity (Embeddings)**
  Uses `sentence-transformers` to understand contextual meaning and related concepts.

* **Skill Extraction & Gap Analysis**
  Identifies:

  * Matched skills
  * Missing skills
    based on a curated technical skill list.

---

### **2. Hybrid Scoring Model**

A final **Job Fit Score (%)** is computed using:

* 50% Semantic similarity
* 30% Skill overlap
* 20% TF-IDF similarity

This balances:

* contextual understanding
* explicit skill requirements
* keyword precision

---

### **3. Interactive UI (Streamlit App)**

A lightweight web interface allows users to:

* Upload a resume (`.txt` or `.pdf`)
* Or paste resume text
* Paste a job description
* Generate:

  * Job Fit Score (%)
  * TF-IDF match score
  * Semantic match score
  * Skill overlap score
  * Matched & missing skills

---

## **Project Structure**

```text
resume_job_matcher/
├── data/
│   ├── jobs.csv
│   └── resume.txt
├── notebooks/
│   └── resume_job_matching.ipynb
├── outputs/
│   ├── final_prototype_results.csv
│   ├── ranked_jobs.csv
│   ├── prototype_results_chart.png
│   └── tfidf_vs_semantic_comparison.csv
├── app.py
├── requirements.txt
└── README.md
```

---

## **Notebook Workflow**

The notebook follows a structured pipeline:

1. Imports and setup
2. Load input data
3. Text preprocessing
4. TF-IDF similarity (baseline model)
5. Skill extraction and gap analysis
6. Prototype scoring and ranking
7. Semantic similarity (advanced model)
8. TF-IDF vs semantic comparison
9. Final hybrid scoring model
10. Evaluation and reflection

---

## **Requirements**

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Main libraries:

* pandas
* nltk
* scikit-learn
* matplotlib
* sentence-transformers
* streamlit
* PyPDF2

---

## **How to Run**

### **Notebook (Analysis & Development)**

1. Open:

   ```
   notebooks/resume_job_matching.ipynb
   ```
2. Run all cells from top to bottom.
3. Ensure data files exist in `data/`.
4. Outputs will be saved in `outputs/`.

---

### **Streamlit App (UI)**

Run:

```bash
streamlit run app.py
```

Then open the browser link (usually `http://localhost:8501`).

---

## **Inputs**

* `data/resume.txt`: Resume used for matching (plain text)
* `data/jobs.csv`: Job dataset with:

  * title
  * company
  * description

### UI Inputs:

* Resume upload (`.txt` or `.pdf`) or pasted text
* Single job description (pasted)

---

## **Outputs**

### Notebook outputs:

* `ranked_jobs.csv`: TF-IDF ranking results
* `final_prototype_results.csv`: Combined prototype scores
* `tfidf_vs_semantic_comparison.csv`: Method comparison
* `prototype_results_chart.png`: Visualization

### UI outputs:

* Job Fit Score (%)
* TF-IDF match
* Semantic match
* Skill overlap
* Matched skills
* Missing skills

---

## **Core Insights**

* **TF-IDF** is precise but limited to exact keywords
* **Semantic similarity** captures meaning but can overgeneralize
* **Skill overlap** improves interpretability but may introduce bias

A **hybrid approach** produces the most balanced results.

---

## **Limitations**

* Scoring weights are manually defined (not learned from real data)
* Skill extraction depends on a predefined list
* Does not account for:

  * experience level
  * years of experience
  * project depth
* Does not predict actual hiring outcomes (only job-fit estimation)

---

## **Future Improvements**

* Learn scoring weights from real hiring data
* Improve skill extraction using NLP (e.g., NER)
* Add resume parsing for structured formats (PDF/DOCX)
* Integrate LLM-based explanations or recommendations
* Expand UI into a full application

---

## **Summary**

This project demonstrates how combining multiple NLP techniques can improve resume-to-job matching by balancing:

* keyword precision
* contextual understanding
* skill-based reasoning

It evolves from a simple prototype into a **hybrid decision-support system** with both analytical and practical applications.

---


