import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_and_evaluate_loso(df: pd.DataFrame) -> dict:
    """
    Trains a Random Forest classifier using Leave-One-Subject-Out (LOSO) Cross-Validation.
    """
    print("Preparing data for LOSO Cross-Validation...")
    
    # 1. Define Features (X), Target (y), and Groups (for LOSO)
    # We drop 'Label' (our target) and 'Subject_ID' (our group tracker) from the features
    feature_cols = [col for col in df.columns if col not in ['Label', 'Subject_ID']]
    
    X = df[feature_cols].values
    y = df['Label'].values
    groups = df['Subject_ID'].values

    # 2. Initialize the Cross-Validator and the Model
    logo = LeaveOneGroupOut()
    
    # Random Forest is highly robust for tabular biometric data
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    scaler = StandardScaler()

    accuracies = []
    f1_scores = []
    
    # We will accumulate feature importances across all folds to find the best signals
    feature_importances = np.zeros(len(feature_cols))

    print(f"Starting Training across {logo.get_n_splits(X, y, groups)} folds (subjects)...")

    # 3. Execute the LOSO Loop
    fold = 1
    for train_idx, test_idx in logo.split(X, y, groups):
        # Split the data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale the features (important for ML models)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Predict on the unseen subject
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics for this fold
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        accuracies.append(acc)
        f1_scores.append(f1)
        feature_importances += model.feature_importances_
        
        # Identify which subject was used as the test set
        test_subject = groups[test_idx][0]
        print(f"Fold {fold} (Test Subject: {test_subject}) - Accuracy: {acc:.2f}, F1: {f1:.2f}")
        fold += 1

    # 4. Aggregate Results
    avg_accuracy = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)
    avg_importances = feature_importances / logo.get_n_splits(X, y, groups)

    # Map the importances back to the feature names
    importance_dict = dict(zip(feature_cols, avg_importances))
    # Sort them from most important to least important
    importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}

    print("\n=====================================")
    print("LOSO CROSS-VALIDATION COMPLETE")
    print(f"Average Accuracy: {avg_accuracy:.3f}")
    print(f"Average F1-Score: {avg_f1:.3f}")
    print("=====================================")

    return {
        'accuracy': avg_accuracy,
        'f1_score': avg_f1,
        'feature_importances': importance_dict
    }