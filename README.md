# 🫀 Wearable Biometrics: Acute Psychosocial Stress Detection

An end-to-end machine learning pipeline that processes raw high-frequency physiological time-series data to predict acute human stress responses.

### 🔬 Core Research Question
*"Can physiological signals from a wearable chest device reliably distinguish acute psychosocial stress from a resting baseline, and which signals contribute most to that prediction?"*

## 🛠️ The Architecture
This project transforms raw 700Hz sensor data into a clean, interactive clinical dashboard through a modular pipeline:
1. **Data Ingestion:** Custom loaders for the WESAD (Wearable Stress and Affect Detection) dataset.
2. **Signal Processing:** Artifact removal and signal detrending using `NeuroKit2`.
3. **Feature Extraction:** Extraction of continuous biomarkers over 60-second sliding windows, including:
   - Heart Rate Variability (HRV - RMSSD, MeanNN)
   - Skin Conductance (EDA - Tonic and Phasic components)
4. **Machine Learning:** A Random Forest classifier strictly validated using **Leave-One-Subject-Out (LOSO) Cross-Validation** to prevent time-series data leakage.
5. **Interactive UI:** A real-time `Streamlit` dashboard for performance evaluation and subject-level visualization.

## 📊 Key Results
By strictly testing the model on unseen subjects (LOSO validation across 15 folds), the pipeline achieved:
* **Average Accuracy:** 87.8%
* **Average F1-Score:** 85.9%

**Top 3 Predictive Biomarkers:**
1. Heart Rate Variability (`HRV_MeanNN`)
2. Average Heart Rate (`Mean_HR`)
3. Background Skin Conductance (`EDA_Tonic_Mean`)

## 🚀 How to Run Locally
1. Clone the repository: `git clone [https://github.com/masum-abid/wesad-stress-detection.git]`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the WESAD dataset and place it in `data/WESAD/`.
4. Run the data processing pipeline: `python test_run.py` *(Note: requires ~15 mins for DSP calculations)*
5. Launch the dashboard: `streamlit run app.py`