# Water Pipe Failure Prediction System

## Project 3B - Company Chameleon

### Team Members
- Rupanshi
- Ashley

### Project Description
Predictive system that identifies water pipes most likely to fail
based on age, material type, soil conditions, and repair history.

### Datasets Used
1. EPA Water Infrastructure Data - data.epa.gov (5,001 records)
2. Syracuse NY Water Main Breaks - data.syrgov.net (1,000 records)

### Best Model
Random Forest - AUC: 0.8468 | F1: 0.6758 | Accuracy: 0.7633

### Risk Assessment Results
- CRITICAL: 656 pipes (21.9%)
- HIGH: 303 pipes (10.1%)
- MEDIUM: 179 pipes (6.0%)
- LOW: 550 pipes (18.3%)
- MINIMAL: 1,312 pipes (43.7%)
- Total Expected Loss: $141,838,239

### Technologies
- Python, XGBoost, Random Forest, Gradient Boosting
- SMOTE, SHAP, Scikit-learn
- RAG with AWWA/EPA infrastructure standards
- Prompt Engineering
- Google Gemini LLM
- Google Colab

### How to Run
1. Open the notebook in Google Colab
2. Run all cells in order
3. Results saved to outputs and reports folders

### References
1. EPA ECHO Federal Database - https://data.epa.gov
2. City of Syracuse Open Data - https://data.syrgov.net
3. AWWA C151, C900, M28 Standards
4. EPA DWINS 6th Assessment Report
5. NACE SP0169 Soil Corrosion Standards
