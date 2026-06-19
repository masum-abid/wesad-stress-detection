import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def train_and_evaluate_loso(df: pd.DataFrame, save_dir: str = None) -> dict:
    """
    Trains a Random Forest using Leave-One-Subject-Out (LOSO) Cross-Validation.
    After evaluation, trains a final model on all data and optionally saves it.
    """
    print("Preparing data for LOSO Cross-Validation...")

    feature_cols = [col for col in df.columns if col not in ['Label', 'Subject_ID']]
    X      = df[feature_cols].values
    y      = df['Label'].values
    groups = df['Subject_ID'].values

    logo     = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)

    accuracies       = []
    f1_scores        = []
    per_subject      = []
    feat_importances = np.zeros(len(feature_cols))

    print(f"Starting LOSO across {n_splits} folds (one per subject)...\n")

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fresh scaler per fold — fit ONLY on training data
        fold_scaler    = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_test_scaled  = fold_scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42,
                                       class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='binary')

        accuracies.append(acc)
        f1_scores.append(f1)
        feat_importances += model.feature_importances_

        test_subject = groups[test_idx][0]
        per_subject.append({
            'subject':  test_subject,
            'accuracy': round(acc, 3),
            'f1':       round(f1, 3),
            'n_windows': len(y_test)
        })

        print(f"  Fold {fold:02d} | Subject {test_subject} | "
              f"Acc: {acc:.3f} | F1: {f1:.3f} | Windows: {len(y_test)}")

    # Aggregate
    avg_accuracy = np.mean(accuracies)
    avg_f1       = np.mean(f1_scores)
    std_accuracy = np.std(accuracies)
    std_f1       = np.std(f1_scores)

    avg_importances = feat_importances / n_splits
    importance_dict = dict(sorted(
        zip(feature_cols, avg_importances),
        key=lambda x: x[1], reverse=True
    ))

    print(f"\n{'='*45}")
    print(f"  LOSO CROSS-VALIDATION COMPLETE")
    print(f"  Accuracy : {avg_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"  F1 Score : {avg_f1:.3f} ± {std_f1:.3f}")
    print(f"{'='*45}")

    # Train final model on ALL data for Streamlit
    print("\nTraining final model on all subjects...")
    final_scaler  = StandardScaler()
    X_all_scaled  = final_scaler.fit_transform(X)
    final_model   = RandomForestClassifier(n_estimators=100, random_state=42,
                                           class_weight='balanced')
    final_model.fit(X_all_scaled, y)
    print("  Final model trained.")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(final_model,  os.path.join(save_dir, 'random_forest.pkl'))
        joblib.dump(final_scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(feature_cols, os.path.join(save_dir, 'feature_names.pkl'))
        print(f"  Model saved to {save_dir}/")

    return {
        'accuracy':            avg_accuracy,
        'accuracy_std':        std_accuracy,
        'f1_score':            avg_f1,
        'f1_std':              std_f1,
        'per_subject':         per_subject,
        'feature_importances': importance_dict,
        'feature_names':       feature_cols,
        'final_model':         final_model,
        'final_scaler':        final_scaler
    }