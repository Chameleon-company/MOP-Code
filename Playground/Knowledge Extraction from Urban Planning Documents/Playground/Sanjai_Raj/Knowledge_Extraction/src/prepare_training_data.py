import pandas as pd
import os

def prepare_data():
    # Load your validated results
    csv_path = 'outputs/extracted_results.csv'
    if not os.path.exists(csv_path):
        print("Error: Run src.main first to generate the CSV!")
        return

    df = pd.read_csv(csv_path)
    
    # We are creating a "training set" by labeling sentences 
    # that actually contain the information we want.
    training_data = []
    
    for index, row in df.iterrows():
        # Labeling Zoning hits
        if row['Zoning'] != "Not Found":
            training_data.append({"text": f"The zone is {row['Zoning']}", "label": "zoning"})
        
        # Labeling Height hits
        if row['Max_Building_Height'] != "Not Found":
            training_data.append({"text": f"The height limit is {row['Max_Building_Height']}", "label": "requirement"})

    # Save for the Machine Learning model
    new_df = pd.DataFrame(training_data)
    new_df.to_csv('outputs/training_data.csv', index=False)
    print("Success: training_data.csv created in outputs/")

if __name__ == "__main__":
    prepare_data()