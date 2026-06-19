# 🫀 Wearable Biometrics: Acute Psychosocial Stress Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-WESAD-orange)

An end-to-end machine learning pipeline that processes raw high-frequency physiological time-series data to predict acute human stress responses.

### Core Research Question
*"Can physiological signals from a wearable chest device reliably distinguish acute psychosocial stress from a resting baseline, and which signals contribute most to that prediction?"*

---

## Pipeline Architecture

This project transforms raw 700Hz sensor data into a clean, interactive clinical dashboard through a modular pipeline:

1. **Data Ingestion** — Custom loaders for the WESAD dataset
2. **Signal Processing** — Artifact removal and detrending via `NeuroKit2`
3. **Feature Extraction** — Biomarkers over 60-second sliding windows:
   - Heart Rate Variability (HRV — RMSSD, MeanNN)
   - Skin Conductance (EDA — Tonic and Phasic components)
4. **Machine Learning** — Random Forest with **Leave-One-Subject-Out (LOSO)** cross-validation to prevent data leakage
5. **Interactive UI** — Streamlit dashboard for performance evaluation and subject-level visualization

---

## Key Results

LOSO validation ensures each subject is tested on a model that has **never seen their data**, giving a realistic measure of generalizability.

| Evaluation | Accuracy | F1 Score | Note |
|---|---|---|---|
| Within-subject (S2 only) | 93.3% | 92.0% | Optimistic — same subject train/test |
| **LOSO cross-subject (15 subjects)** | **87.8%** | **85.9%** | Realistic — unseen subject evaluation |

The gap between the two rows is the core scientific argument of this project: single-subject evaluation significantly overestimates real-world performance.

**Top 3 Predictive Biomarkers (averaged across all 15 LOSO folds):**
1. Heart Rate Variability (`HRV_MeanNN`)
2. Average Heart Rate (`Mean_HR`)
3. Background Skin Conductance (`EDA_Tonic_Mean`)

---

## Repository Structure
```
├── notebooks/
│ ├── 01_exploration.ipynb # Raw signal inspection (S2)
│ ├── 02_visualization.ipynb # Stress vs baseline visualizations (S2)
│ ├── 03_features.ipynb # Feature extraction & distributions (S2)
│ ├── 04_modeling.ipynb # First model — single-subject baseline (S2)
│ ├── 05_explainability.ipynb # SHAP feature importance (S2)
│ └── 06_multisubject.ipynb # LOSO evaluation across all 15 subjects ⭐
├── src/
│ ├── processing.py # Signal preprocessing
│ ├── features.py # Feature extraction
│ └── models.py # LOSO training & evaluation
├── outputs/ # Generated figures
├── data/ # WESAD dataset (not included, see below)
├── app.py # Streamlit dashboard
└── requirements.txt
```

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/masum-abid/wesad-stress-detection.git
   cd wesad-stress-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the [WESAD dataset](https://archive.ics.uci.edu/dataset/465/wesad) and place it in `data/WESAD/`

4. Run the processing pipeline:
   ```bash
   python run_pipeline.py
   ```
   *(Note: requires ~15 mins for full DSP calculations across 15 subjects)*

5. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

---

## Reference

Schmidt, P., et al. (2018). *WESAD: A Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI 2018. [https://doi.org/10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985)