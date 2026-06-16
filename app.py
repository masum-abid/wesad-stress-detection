import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import train_and_evaluate_loso

# --- Page Configuration ---
st.set_page_config(page_title="Stress Detection Pipeline", page_icon="🫀", layout="wide")

# --- Data Loading & Caching ---
# We use st.cache_data so the model doesn't retrain every time the user clicks a button!
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/wesad_features_master.csv')
    return df.dropna()

@st.cache_data
def run_model(_df):
    return train_and_evaluate_loso(_df)

# --- App Layout ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/FAU_Logo.svg/512px-FAU_Logo.svg.png", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Model Performance", "Subject Analysis"])

st.title("Wearable Biometrics: Acute Stress Detection")
st.markdown("*An end-to-end machine learning pipeline for processing time-series physiological data.*")

# Load data into memory
df = load_data()

if page == "Project Overview":
    st.header("🔬 Research Question")
    st.info("**Can physiological signals from a wearable chest device reliably distinguish acute psychosocial stress from a resting baseline, and which signals contribute most to that prediction?**")
    
    st.write("### The Dataset: WESAD")
    st.write("This pipeline utilizes the WESAD (Wearable Stress and Affect Detection) dataset. We extracted raw 700Hz chest sensor data (ECG, EDA, Respiration, Temperature) from 15 subjects.")
    
    st.write("### The Engineering Pipeline")
    st.markdown("""
    1. **Signal Processing:** Filtered raw voltage signals using `NeuroKit2`.
    2. **Feature Extraction:** Extracted Heart Rate Variability (HRV) and Tonic/Phasic skin conductance using a 60-second sliding window.
    3. **Machine Learning:** Trained a Random Forest classifier using Leave-One-Subject-Out (LOSO) Cross-Validation to prevent data leakage.
    """)
    
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Model Performance":
    st.header("📊 Model Evaluation (LOSO Cross-Validation)")
    
    with st.spinner("Training model across 15 folds..."):
        results = run_model(df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Accuracy", f"{results['accuracy'] * 100:.1f}%")
    col2.metric("Average F1-Score", f"{results['f1_score'] * 100:.1f}%")
    col3.metric("Validation Strategy", "LOSO (15 Folds)")
    
    st.write("---")
    st.subheader("Feature Importance")
    st.write("Which physiological signals provided the highest predictive power?")
    
    # Plot feature importances
    importances = results['feature_importances']
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(importances.values()), y=list(importances.keys()), palette="viridis", ax=ax)
    ax.set_xlabel("Mean Decrease in Impurity (Random Forest)")
    ax.set_title("Signal Contribution to Stress Detection")
    st.pyplot(fig)

elif page == "Subject Analysis":
    st.header("👤 Individual Subject Visualization")
    
    subjects = df['Subject_ID'].unique()
    selected_subject = st.selectbox("Select a Subject to View", subjects)
    
    subj_df = df[df['Subject_ID'] == selected_subject].reset_index()
    
    st.write(f"### Physiological Response over Time for {selected_subject}")
    
    # Create a 2-row plot: HR on top, EDA on bottom, colored by stress state
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Scatter plot for Heart Rate
    sns.scatterplot(data=subj_df, x=subj_df.index, y='Mean_HR', hue='Label', 
                    palette={0: 'blue', 1: 'red'}, ax=ax1, legend=False)
    ax1.set_ylabel("Heart Rate (BPM)")
    ax1.set_title("Heart Rate Response (Blue = Baseline, Red = Stress)")
    
    # Scatter plot for EDA Tonic
    sns.scatterplot(data=subj_df, x=subj_df.index, y='EDA_Tonic_Mean', hue='Label', 
                    palette={0: 'blue', 1: 'red'}, ax=ax2)
    ax2.set_ylabel("Skin Conductance (Tonic)")
    ax2.set_xlabel("Time (Minutes)")
    
    # Clean up legend
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, ['Baseline', 'Stress'], loc='upper right')
    
    st.pyplot(fig)