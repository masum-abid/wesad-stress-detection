import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import train_and_evaluate_loso

st.set_page_config(page_title="Stress Detection Pipeline", page_icon="🫀", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data/features_all_subjects.csv')
    return df.dropna()

@st.cache_resource
def run_model():
    df = load_data()
    return train_and_evaluate_loso(df)

st.sidebar.markdown("## 🫀 Stress Detection")
st.sidebar.markdown("*WESAD · 15 Subjects · LOSO CV*")
st.sidebar.divider()
page = st.sidebar.radio("Navigation", ["Project Overview", "Model Performance", "Subject Analysis"])

st.title("Wearable Biometrics: Acute Stress Detection")
st.markdown("*An end-to-end machine learning pipeline for processing time-series physiological data.*")

df = load_data()

if page == "Project Overview":
    st.header("🔬 Research Question")
    st.info("**Can physiological signals from a wearable chest device reliably distinguish acute psychosocial stress from a resting baseline, and which signals contribute most to that prediction?**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Subjects", "15")
    col2.metric("Validation", "LOSO CV")
    col3.metric("Window Size", "60s")

    st.write("### The WESAD Dataset")
    st.write("Raw 700Hz chest sensor data (ECG, EDA, Respiration, Temperature) from 15 subjects undergoing the Trier Social Stress Test (TSST).")

    st.write("### Pipeline")
    st.markdown("""
    1. **Signal Processing** — Filtered raw voltage signals using `NeuroKit2`
    2. **Feature Extraction** — HRV and Tonic/Phasic skin conductance over 60-second sliding windows
    3. **Machine Learning** — Random Forest with LOSO cross-validation to prevent data leakage
    """)
    st.write("### Feature Matrix Preview")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Model Performance":
    st.header("📊 Model Evaluation (LOSO Cross-Validation)")

    with st.spinner("Running LOSO across 15 folds — this may take a moment..."):
        results = run_model()

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Accuracy", f"{results['accuracy']*100:.1f}%")
    col2.metric("Average F1-Score", f"{results['f1_score']*100:.1f}%")
    col3.metric("Validation Strategy", "LOSO (15 Folds)")

    st.write("---")
    st.subheader("📌 Why LOSO? The Generalization Gap")
    comparison_df = pd.DataFrame({
        'Evaluation Type': ['Within-Subject (S2 only)', 'LOSO Cross-Subject (15 subjects)'],
        'Accuracy': ['93.3%', f"{results['accuracy']*100:.1f}%"],
        'F1 Score': ['92.0%', f"{results['f1_score']*100:.1f}%"],
        'Realistic?': ['❌ Overoptimistic', '✅ Generalizable']
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.caption("Training and testing on the same subject inflates performance. LOSO ensures each subject is tested on a model that has never seen their data.")

    st.write("---")
    st.subheader("Feature Importance")
    importances = results['feature_importances']
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(importances.values()), y=list(importances.keys()), palette="viridis", ax=ax)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Signal Contribution to Stress Detection")
    st.pyplot(fig)

    st.write("---")
    st.subheader("Per-Subject LOSO Accuracy")
    per_subject_df = pd.DataFrame(results['per_subject']).sort_values('accuracy')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=per_subject_df, x='subject', y='accuracy', palette='Blues_d', ax=ax)
    ax.axhline(results['accuracy'], color='red', linestyle='--', label=f"Mean: {results['accuracy']:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Subject ID")
    ax.set_title("LOSO Accuracy per Subject")
    ax.legend()
    st.pyplot(fig)

elif page == "Subject Analysis":
    st.header("👤 Individual Subject Visualization")

    subjects = sorted(df['Subject_ID'].unique())
    selected_subject = st.selectbox("Select a Subject", subjects)
    subj_df = df[df['Subject_ID'] == selected_subject].reset_index(drop=True)

    with st.spinner("Running LOSO..."):
        results = run_model()

    per_subject_df = pd.DataFrame(results['per_subject'])
    subj_row = per_subject_df[per_subject_df['subject'] == selected_subject]

    if not subj_row.empty:
        subj_acc = subj_row.iloc[0]['accuracy']
        st.metric(
            f"LOSO Accuracy for {selected_subject}",
            f"{subj_acc*100:.1f}%",
            delta=f"{(subj_acc - results['accuracy'])*100:.1f}% vs average"
        )

    st.write(f"### Physiological Response over Time — {selected_subject}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    sns.scatterplot(data=subj_df, x=subj_df.index, y='Mean_HR', hue='Label',
                    palette={0: '#4682b4', 1: '#e05c5c'}, ax=ax1, legend=False, s=15)
    ax1.set_ylabel("Heart Rate (BPM)")
    ax1.set_title("Heart Rate  (Blue = Baseline · Red = Stress)")

    sns.scatterplot(data=subj_df, x=subj_df.index, y='EDA_Tonic_Mean', hue='Label',
                    palette={0: '#4682b4', 1: '#e05c5c'}, ax=ax2, s=15)
    ax2.set_ylabel("Skin Conductance (Tonic)")
    ax2.set_xlabel("Window Index")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, ['Baseline', 'Stress'], loc='upper right')

    plt.tight_layout()
    st.pyplot(fig)