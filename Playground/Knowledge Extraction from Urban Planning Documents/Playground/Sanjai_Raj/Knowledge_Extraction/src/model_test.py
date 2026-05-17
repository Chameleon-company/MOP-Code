import spacy
import pandas as pd

# Load the Australian English compatible model
nlp = spacy.load("en_core_web_sm")

def generate_model_insights(text):
    # Process the text through the NLP pipeline
    doc = nlp(text)
    
    # Extract entities like Organisations and Dates
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Test the model on a small snippet from your urban documents
sample_text = "The City of Melbourne requires a permit for manufacturing in this zone as of 2026."
print("--- Model Generation Output ---")
print(generate_model_insights(sample_text))