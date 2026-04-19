import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
import os

def load_pmc_data(mat_file_path):
    """
    Load the PMC dataset from a .mat file.
    Returns a dictionary with sensor data and metadata.
    """
    mat_data = sio.loadmat(mat_file_path)
    
    # The structure may vary; inspect keys to find acceleration data
    print("Keys in .mat file:", mat_data.keys())
    
    # Typically, the acceleration data is under 'data' or 'acc'
    # This is an example; you'll need to adjust based on actual keys.
    if 'acc' in mat_data:
        acc_data = mat_data['acc']  # shape: (time_steps, num_sensors)
    else:
        # Fallback: try to find any array that looks like sensor data
        for key in mat_data:
            if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 2:
                acc_data = mat_data[key]
                break
        else:
            raise ValueError("Could not locate acceleration data.")
    
    # Create a DataFrame for easier handling
    df = pd.DataFrame(acc_data, columns=[f'sensor_{i}' for i in range(acc_data.shape[1])])
    
    # Assume sampling frequency is known (e.g., 100 Hz)
    # You can add a time index if needed
    return df

# Add this function
def load_synthetic_data():
    np.random.seed(42)
    data = np.random.randn(10000, 8) * 0.01
    return pd.DataFrame(data, columns=[f'sensor_{i}' for i in range(data.shape[1])])

if __name__ == "__main__":
    # Replace the file loading with synthetic
    df = load_synthetic_data()
    print(df.head())
    print(f"Loaded data shape: {df.shape}")