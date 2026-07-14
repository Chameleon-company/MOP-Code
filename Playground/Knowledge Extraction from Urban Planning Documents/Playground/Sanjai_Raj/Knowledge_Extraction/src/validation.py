import pandas as pd
import os

def run_validation():
    csv_path = 'outputs/extracted_results.csv'
    
    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: {csv_path} does not exist. Run 'python -m src.main' first!")
        return

    # Load the data
    df = pd.read_csv(csv_path)
    
    # DEBUG: Let's see what was actually extracted
    print("\n--- DEBUG: WHAT THE AI FOUND ---")
    print(df[['document_name', 'Zoning']].to_string())
    print("--------------------------------\n")

    # The most flexible list of valid Victorian terms
    valid_keywords = [
        'C1Z', 'GRZ', 'NRZ', 'TZ', 'RGZ', 'B1Z', 'IN1Z', 
        'Township', 'Urban', 'Growth', 'Policy', 'Area', 
        'Residential', 'Industrial', 'Rural', 'Conservation', 'Railway'
    ]
    
    def check_validity(val):
        val_str = str(val).strip()
        if val_str == "Not Found" or val_str == "" or val_str == "nan":
            return False
        # Check if ANY of our keywords are in the extracted text
        return any(key.lower() in val_str.lower() for key in valid_keywords)

    df['Zoning_Valid'] = df['Zoning'].apply(check_validity)
    
    total_docs = len(df)
    valid_hits = df['Zoning_Valid'].sum()
    
    # Calculate accuracy
    accuracy = (valid_hits / total_docs) * 100 if total_docs > 0 else 0

    print("========================================")
    print("       FINAL VALIDATION REPORT")
    print("========================================")
    print(f"Total Documents:      {total_docs}")
    print(f"Valid Zoning Hits:    {valid_hits}")
    print(f"Accuracy Score:       {accuracy:.2f}%")
    print("========================================")

if __name__ == "__main__":
    run_validation()