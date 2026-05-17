import joblib
import os

def predict_sentence(text):
    # 1. Load the Brain and the Translator
    if not os.path.exists('outputs/planning_model.pkl'):
        return "Model not trained yet."
    
    model = joblib.load('outputs/planning_model.pkl')
    vectorizer = joblib.load('outputs/vectorizer.pkl')

    # 2. Transform the new text and predict
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    
    return prediction[0]

if __name__ == "__main__":
    # Test it with a sentence the Regex might struggle with
    test_text = "The building scale must align with the township character and not exceed the tree canopy."
    result = predict_sentence(test_text)
    print(f"Input: {test_text}")
    print(f"AI Classification: {result}")