import os
import pickle
import numpy as np
import pandas as pd

class WESADDataset:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_subject(self, subject_id: str) -> pd.DataFrame:
        file_path = os.path.join(self.data_path, subject_id, f"{subject_id}.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file for {subject_id} not found at {file_path}")
        print(f"Loading {subject_id} data... This might take a moment.")
        
        # WESAD was created in Python 2, so we MUST use encoding='latin1' to read it in Python 3.
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        
        chest_signals = data['signal']['chest']
        labels = data['label']

        # Flatten the arrays from shape (N, 1) to (N,)
        df = pd.DataFrame({
            'ECG': chest_signals['ECG'].flatten(),
            'EDA': chest_signals['EDA'].flatten(),
            'EMG': chest_signals['EMG'].flatten(),
            'Resp': chest_signals['Resp'].flatten(),
            'Temp': chest_signals['Temp'].flatten(),
            'Label': labels.flatten()
        })

        return df
    
    def filter_target_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        We only care about classifying 'Baseline' (1) vs 'Stress' (2).
        This function removes all other transitional or amusement states.
        """
        # Keep only Baseline (1) and Stress (2)
        filtered_df = df[df['Label'].isin([1, 2])].copy()
        
        # Optional: Remap Baseline to 0 and Stress to 1 for standard binary classification
        filtered_df['Label'] = filtered_df['Label'].map({1: 0, 2: 1})
        
        return filtered_df