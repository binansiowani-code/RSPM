import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

def load_and_preprocess_data(filepath=None):
    if filepath is None:
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'rspm_integrated.csv'))
    df = pd.read_csv(filepath)
    
    # Data Cleaning and Missing values (though our synthetic data has none, we handle it as requested)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Separation of features and targets
    # We will exclude variables that directly give away the exact classification easily like "Failure_Probability"
    # as we want the models to learn from raw inputs. 
    # But as per prompt "Use calculated engineering outputs as additional ML features": 
    # we will keep 'Calculated_Failure_Pressure_psi' and other physical features.
    
    features = [
        'Reservoir_Pressure_psi', 'Reservoir_Temperature_C', 'Oil_Production_Rate_bbl_day',
        'Gas_Production_Rate_MSCF_day', 'Water_Cut_percent', 'Pipeline_Diameter_m',
        'Wall_Thickness_mm', 'Pipeline_Length_km', 'Flow_Velocity_m_s', 'Fluid_Density_kg_m3',
        'Fluid_Viscosity_cP', 'Corrosion_Rate_mm_year', 'Internal_Pressure_psi',
        'Temperature_Gradient_C_km', 'Elevation_Change_m', 'Pipeline_Age_years',
        'Calculated_Failure_Pressure_psi'
    ]
    
    X = df[features]
    y = df['Leak_Risk_Class']
    
    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference
    # Now deferred to save together with model
    
    return X_train_scaled, X_test_scaled, y_train, y_test, features, scaler

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    # Using weighted metrics since classes might not be perfectly balanced
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred, labels=['Low', 'Medium', 'High'])
    
    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")
    return accuracy

def train_models():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()
    
    # Train Model 1: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_accuracy = evaluate_model(y_test, rf_preds, "Random Forest")
    
    # Train Model 2: SVM
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    svm_accuracy = evaluate_model(y_test, svm_preds, "Support Vector Machine (SVM)")
    
    # Compare and save the best model
    best_model = rf_model if rf_accuracy >= svm_accuracy else svm_model
    model_name = "Random Forest" if rf_accuracy >= svm_accuracy else "SVM"
    print(f"{model_name} is selected as the best model.")
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'rspm_models.pkl'))
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names
    }, model_path)
    
    print("Best model, scaler, and features saved successfully.")

if __name__ == "__main__":
    train_models()
