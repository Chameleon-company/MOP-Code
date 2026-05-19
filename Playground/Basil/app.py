import streamlit as st
import re
import string
import nltk
import PyPDF2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Resume-to-Job Matching Assistant", layout="centered")

st.title("Resume-to-Job Matching Assistant")
st.write("Upload a resume or paste resume text, then paste a job description to get a job-fit score.")

# -----------------------------
# NLTK setup
# -----------------------------
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt")
    nltk.download("stopwords")

download_nltk_resources()

stop_words = set(stopwords.words("english"))

# -----------------------------
# Skill list
# -----------------------------
skill_list = [
    "python", "sql", "excel", "tableau", "power bi", "machine learning",
    "deep learning", "data analysis", "data visualization", "nlp",
    "computer vision", "opencv", "yolo", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "git", "aws",
    "feature engineering", "model evaluation", "dashboard", "reporting",
    "statistics", "data preprocessing", "business intelligence"
]

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

def extract_skills(text: str, skills: list[str]) -> list[str]:
    text = text.lower()
    found_skills = []
    for skill in skills:
        if skill in text:
            found_skills.append(skill)
    return sorted(set(found_skills))

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

semantic_model = load_semantic_model()

def compute_tfidf_score(resume_text: str, job_text: str) -> float:
    cleaned_resume = preprocess_text(resume_text)
    cleaned_job = preprocess_text(job_text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_job])

    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)

def compute_semantic_score(resume_text: str, job_text: str) -> float:
    resume_embedding = semantic_model.encode([resume_text])
    job_embedding = semantic_model.encode([job_text])

    score = cosine_similarity(resume_embedding, job_embedding)[0][0]
    return round(score * 100, 2)

def compute_skill_scores(resume_text: str, job_text: str):
    resume_skills = extract_skills(resume_text, skill_list)
    job_skills = extract_skills(job_text, skill_list)

    matched_skills = sorted(set(resume_skills).intersection(set(job_skills)))
    missing_skills = sorted(set(job_skills) - set(resume_skills))

    if len(job_skills) == 0:
        skill_overlap_score = 0.0
    else:
        skill_overlap_score = round((len(matched_skills) / len(job_skills)) * 100, 2)

    return resume_skills, job_skills, matched_skills, missing_skills, skill_overlap_score

def compute_final_score(tfidf_score: float, semantic_score: float, skill_overlap_score: float) -> float:
    final_score = (
        0.2 * tfidf_score +
        0.5 * semantic_score +
        0.3 * skill_overlap_score
    )
    return round(final_score, 2)


def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    
    for page in reader.pages:
        try:
            text += page.extract_text() + "\n"
        except:
            pass
    
    return text
# -----------------------------
# Inputs
# -----------------------------
st.subheader("Resume Input")
uploaded_file = st.file_uploader("Upload resume (.txt or .pdf)", type=["txt", "pdf"])
resume_text_input = st.text_area("Or paste resume text here", height=200)

st.subheader("Job Description Input")
job_description = st.text_area("Paste the job description here", height=250)

# -----------------------------
# Resolve resume text
# -----------------------------
resume_text = ""

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        resume_text = uploaded_file.read().decode("utf-8")
    
    elif uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
if uploaded_file is not None and not resume_text.strip():
    st.warning("Could not extract text from the uploaded file. Try a different PDF or paste text manually.")

# -----------------------------
# Analyze button
# -----------------------------
if st.button("Analyze Match"):
    if not resume_text.strip():
        st.error("Please upload a resume file or paste resume text.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        tfidf_score = compute_tfidf_score(resume_text, job_description)
        semantic_score = compute_semantic_score(resume_text, job_description)
        resume_skills, job_skills, matched_skills, missing_skills, skill_overlap_score = compute_skill_scores(
            resume_text, job_description
        )
        final_score = compute_final_score(tfidf_score, semantic_score, skill_overlap_score)

        st.success("Analysis complete.")

        st.subheader("Results")
        st.metric("Job Fit Score", f"{final_score}%")
        st.metric("TF-IDF Match", f"{tfidf_score}%")
        st.metric("Semantic Match", f"{semantic_score}%")
        st.metric("Skill Overlap", f"{skill_overlap_score}%")

        st.subheader("Skill Analysis")
        st.write("**Matched Skills:**", ", ".join(matched_skills) if matched_skills else "None")
        st.write("**Missing Skills:**", ", ".join(missing_skills) if missing_skills else "None")

        st.subheader("Interpretation")
        if final_score >= 70:
            st.write("This role appears to be a strong match based on resume-job alignment.")
        elif final_score >= 50:
            st.write("This role appears to be a moderate match, but some skill gaps exist.")
        else:
            st.write("This role appears to be a weaker match based on the current resume content.")