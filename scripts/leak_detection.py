import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.architecture_engine import (
    INTEGRATED_FEATURES,
    RAW_INPUT_FEATURES,
    RISK_ORDER,
    augment_with_architecture,
)


def load_and_preprocess_data(filepath=None):
    if filepath is None:
        filepath = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "rspm_integrated.csv")
        )

    df = pd.read_csv(filepath)
    df.fillna(df.median(numeric_only=True), inplace=True)
    augmented = augment_with_architecture(df)

    X = augmented[INTEGRATED_FEATURES]
    y = augmented["Architecture_Risk_Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return augmented, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_classifier(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
    }


def train_models():
    print("Preparing architecture-aligned dataset...")
    augmented, X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_classifier(rf_model, X_test, y_test)
    rf_metrics["cv_accuracy"] = float(
        cross_val_score(rf_model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    )

    svm_model = SVC(kernel="rbf", probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_metrics = evaluate_classifier(svm_model, X_test, y_test)
    svm_metrics["cv_accuracy"] = float(
        cross_val_score(svm_model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    )

    best_model = rf_model if rf_metrics["accuracy"] >= svm_metrics["accuracy"] else svm_model
    best_model_name = "Random Forest" if best_model is rf_model else "SVM RBF"

    X_all = scaler.transform(augmented[INTEGRATED_FEATURES])
    lrs_regressor = RandomForestRegressor(n_estimators=250, random_state=42)
    lrs_regressor.fit(X_all, augmented["Architecture_LRS"])

    corrosion_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
    corrosion_regressor.fit(X_all, augmented["Corrosion_Rate_mm_year"])

    failure_pressure_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
    failure_pressure_regressor.fit(X_all, augmented["Calculated_Failure_Pressure_psi"])

    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "rspm_models.pkl")
    )
    joblib.dump(
        {
            "architecture_version": "layered-v1",
            "model": best_model,
            "model_name": best_model_name,
            "scaler": scaler,
            "feature_names": INTEGRATED_FEATURES,
            "raw_input_features": RAW_INPUT_FEATURES,
            "risk_order": RISK_ORDER,
            "metrics": {
                "random_forest": rf_metrics,
                "svm_rbf": svm_metrics,
            },
            "regressors": {
                "lrs": lrs_regressor,
                "corrosion_rate": corrosion_regressor,
                "failure_pressure": failure_pressure_regressor,
            },
        },
        model_path,
    )

    print(f"Saved architecture-compliant bundle to {model_path}")
    print(f"Selected classifier: {best_model_name}")


if __name__ == "__main__":
    train_models()
