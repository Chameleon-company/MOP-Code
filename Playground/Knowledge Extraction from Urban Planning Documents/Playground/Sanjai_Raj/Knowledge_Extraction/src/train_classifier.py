import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model():
    # 1. Load the data you just prepared
    data_path = data_path = 'outputs/augmented_training_data.csv'
    if not os.path.exists(data_path):
        print("Error: No training data found. Run prepare_training_data first!")
        return

    df = pd.read_csv(data_path)
    
    if len(df) < 2:
        print("Error: Not enough data to train. Need at least 2 different labels.")
        return

    # 2. Text processing: Turning words into numbers (TF-IDF)
    # This helps the AI understand which words are "important" (like 'Zone' or 'Limit')
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # 3. The Classifier: Random Forest
    # It builds multiple 'decision trees' to make a final prediction
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 4. Save the 'Intelligence' (Model) and the 'Translator' (Vectorizer)
    # We need both to make predictions later
    joblib.dump(model, 'outputs/planning_model.pkl')
    joblib.dump(vectorizer, 'outputs/vectorizer.pkl')
    
    print("========================================")
    print("   MACHINE LEARNING TRAINING COMPLETE   ")
    print("========================================")
    print("Model saved as: outputs/planning_model.pkl")
    print("Vectorizer saved as: outputs/vectorizer.pkl")

if __name__ == "__main__":
    train_model()