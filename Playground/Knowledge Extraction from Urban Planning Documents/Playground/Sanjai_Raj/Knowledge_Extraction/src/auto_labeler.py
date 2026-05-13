import os
import pandas as pd

def auto_label_sentences():
    raw_data_path = 'outputs/extracted_results.csv'
    if not os.path.exists(raw_data_path):
        return

    df = pd.read_csv(raw_data_path)
    augmented_data = []

    for index, row in df.iterrows():
        # FIX: We only want labels that actually contain a Planning Keyword
        # This prevents words like 'shall' or 'must' from being the only data
        zoning_val = str(row['Zoning'])
        
        # Check if the extracted text looks like a real zone or policy
        if any(key in zoning_val.lower() for key in ['policy', 'area', 'township', 'urban', 'growth']):
             augmented_data.append({"text": f"The land is located in {zoning_val}", "label": "zoning"})
        
        # Add high-quality requirement examples
        if row['Max_Building_Height'] != "Not Found":
            augmented_data.append({"text": f"The maximum building height limit is {row['Max_Building_Height']}", "label": "requirement"})

    new_df = pd.DataFrame(augmented_data)
    new_df.to_csv('outputs/augmented_training_data.csv', index=False)
    print(f"Refined {len(new_df)} high-quality labels.")

if __name__ == "__main__":
    auto_label_sentences()