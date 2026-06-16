import pandas as pd
import numpy as np
import neurokit2 as nk

def extract_features_from_window(window_df: pd.DataFrame, sampling_rate: int = 700) -> dict:
    """
    Extracts physiological features from a time window of raw chest sensor data.
    """
    features = {}
    
    # 1. Process ECG to get Heart Rate and HRV features
    try:
        # Clean ECG and find peaks
        ecg_cleaned = nk.ecg_clean(window_df['ECG'], sampling_rate=sampling_rate)
        peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        
        # Calculate HRV (Time-domain)
        hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate)
        features['HRV_RMSSD'] = hrv_time['HRV_RMSSD'].values[0]
        features['HRV_MeanNN'] = hrv_time['HRV_MeanNN'].values[0]
        
        # Calculate average Heart Rate for the window
        rate = nk.signal_rate(peaks, sampling_rate=sampling_rate, desired_length=len(window_df))
        features['Mean_HR'] = np.mean(rate)
    except Exception:
        # If the signal is too noisy to find peaks, return NaNs
        features['HRV_RMSSD'] = np.nan
        features['HRV_MeanNN'] = np.nan
        features['Mean_HR'] = np.nan

    # 2. Process EDA (Skin Conductance)
    try:
        # Clean EDA and decompose into Phasic (responses) and Tonic (level)
        eda_cleaned = nk.eda_clean(window_df['EDA'], sampling_rate=sampling_rate)
        eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate)
        
        features['EDA_Tonic_Mean'] = np.mean(eda_decomposed['EDA_Tonic'])
        features['EDA_Phasic_Mean'] = np.mean(eda_decomposed['EDA_Phasic'])
    except Exception:
        features['EDA_Tonic_Mean'] = np.nan
        features['EDA_Phasic_Mean'] = np.nan

    # 3. Simple features for Respiration and Temperature
    features['Temp_Mean'] = np.mean(window_df['Temp'])
    features['Resp_Std'] = np.std(window_df['Resp']) # Standard deviation of breathing
    
    # Label for the window (assuming the window is mostly one state, we take the mode)
    features['Label'] = window_df['Label'].mode()[0]

    return features

def process_subject_data(df: pd.DataFrame, window_size_sec: int = 60, sampling_rate: int = 700) -> pd.DataFrame:
    """
    Slides a window across the raw dataframe and extracts features for each window.
    """
    window_samples = window_size_sec * sampling_rate
    extracted_data = []

    print(f"Extracting features using {window_size_sec}-second windows...")
    
    # Slide across the data in non-overlapping windows
    for start_idx in range(0, len(df), window_samples):
        end_idx = start_idx + window_samples
        
        # Make sure we have a full window
        if end_idx <= len(df):
            window_df = df.iloc[start_idx:end_idx]
            window_features = extract_features_from_window(window_df, sampling_rate)
            extracted_data.append(window_features)

    # Convert the list of feature dictionaries into a new ML-ready DataFrame
    feature_df = pd.DataFrame(extracted_data)
    
    # Drop any windows where the signal was too noisy to process
    feature_df = feature_df.dropna()
    
    return feature_df