# Resume Job Matcher

A notebook-based prototype that ranks job postings against a resume using a combination of text similarity and skill overlap.

## What It Does

This project compares a resume with a set of job descriptions and produces ranked matches. It uses two signals:

- TF-IDF + cosine similarity for overall text relevance
- Skill extraction and overlap for a more targeted match score

The notebook then combines those scores into a final prototype score and exports the results.

## Project Structure

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
│   └── prototype_results_chart.png
├── src/
└── requirements.txt
```

## Notebook Workflow

The notebook is organized into clear sections:

1. Imports and setup
2. Load input data
3. Text preprocessing
4. Text similarity matching
5. Skill extraction and overlap
6. Final scoring and ranking
7. Reporting, visualization, and export

## Requirements

Install the Python packages listed in `requirements.txt`.

Typical dependencies include:

- pandas
- nltk
- scikit-learn
- matplotlib

## How to Run

1. Open `notebooks/resume_job_matching.ipynb`.
2. Run the cells from top to bottom.
3. Make sure the data files are available in `data/`.
4. Review the ranked matches and generated outputs in `outputs/`.

## Inputs

- `data/resume.txt`: Plain-text resume used as the matching profile
- `data/jobs.csv`: Job postings dataset with job titles, companies, and descriptions

## Outputs

The notebook writes the following files to `outputs/`:

- `ranked_jobs.csv`: Jobs ranked by similarity score
- `final_prototype_results.csv`: Final combined results with match and skill scores
- `prototype_results_chart.png`: Bar chart of final prototype scores

## Matching Logic

The ranking is based on a weighted score:

- 70% text match score
- 30% skill overlap score

This keeps the prototype simple while still reflecting both broad relevance and concrete skill alignment.

## Notes

- The notebook downloads the NLTK `punkt` and `stopwords` resources on first run.
- If you change the resume or job data, rerun the notebook from the top to refresh all outputs.

