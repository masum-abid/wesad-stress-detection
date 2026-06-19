import pandas as pd
from src.models import train_and_evaluate_loso

if __name__ == "__main__":
    # Load the Golden Dataset
    print("Loading processed features dataset...")
    df = pd.read_csv('data/features_all_subjects.csv')
    
    # Drop rows with NaN values (sometimes generated if a 60s window was too noisy)
    df = df.dropna()
    
    # Run the ML pipeline
    results = train_and_evaluate_loso(df)
    
    print("\n--- Top 3 Most Important Features ---")
    top_3 = list(results['feature_importances'].items())[:3]
    for feature, importance in top_3:
        print(f"{feature}: {importance:.3f}")