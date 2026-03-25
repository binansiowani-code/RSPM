import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.pipeline_model import calculate_failure_pressure, calculate_failure_probability, determine_pipeline_health_status
from scripts.report_generator import generate_excel_report, generate_pdf_report

# Setup Page Configuration
st.set_page_config(page_title="RSPM Dashboard", layout="wide")

# Load Machine Learning Artifacts
@st.cache_resource
def load_ml_assets():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'rspm_models.pkl'))
    data = joblib.load(model_path)
    model = data['model']
    scaler = data['scaler']
    feature_names = data['feature_names']
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_ml_assets()
except Exception as e:
    st.error(f"Error loading AI models. Please ensure `train_models.py` has been executed. ({e})")
    st.stop()

# Dashboard Title
st.title("🛢️ RSPM: Reservoir & Surface Pipeline Monitoring System")
st.markdown("Predictive pipeline maintenance, physical failure estimation, and intelligent classifications driven by ML.")

# Sidebar Feature Modifiers
st.sidebar.header("Pipeline Operations Inputs")

res_pressure = st.sidebar.slider("Reservoir Pressure (psi)", 3000.0, 8000.0, 4500.0)
res_temp = st.sidebar.slider("Reservoir Temperature (°C)", 60.0, 150.0, 85.0)
oil_rate = st.sidebar.slider("Oil Production Rate (bbl/day)", 1000.0, 50000.0, 15000.0)
gas_rate = st.sidebar.slider("Gas Production Rate (MSCF/day)", 500.0, 20000.0, 8000.0)
water_cut = st.sidebar.slider("Water Cut (%)", 0.0, 90.0, 30.0)

st.sidebar.markdown("---")
st.sidebar.header("Pipeline Engineering Values")
pipeline_diameter = st.sidebar.slider("Pipeline Diameter (m)", 0.1, 1.0, 0.4)
wall_thickness = st.sidebar.slider("Wall Thickness (mm)", 5.0, 25.0, 10.0)
pipeline_length = st.sidebar.slider("Pipeline Length (km)", 5.0, 100.0, 35.0)
flow_velocity = st.sidebar.slider("Flow Velocity (m/s)", 0.5, 5.0, 2.5)
fluid_density = st.sidebar.slider("Fluid Density (kg/m³)", 700.0, 1000.0, 850.0)
fluid_viscosity = st.sidebar.slider("Fluid Viscosity (cP)", 1.0, 50.0, 12.0)

st.sidebar.markdown("---")
st.sidebar.header("Operational Integrity Factors")
corrosion_rate = st.sidebar.slider("Corrosion Rate (mm/year)", 0.01, 2.0, 0.25)
internal_pressure = st.sidebar.slider("Internal Pressure (psi)", 500.0, 3000.0, 1200.0)
temp_gradient = st.sidebar.slider("Temperature Gradient (°C/km)", 0.5, 5.0, 1.5)
elevation_change = st.sidebar.slider("Elevation Change (m)", -200.0, 200.0, -10.0)
pipeline_age = st.sidebar.slider("Pipeline Age (years)", 1.0, 40.0, 15.0)

# Physical Calculations
st.header("1. Pipeline Engineering Analysis")
col1, col2 = st.columns(2)

failure_pressure = calculate_failure_pressure(pipeline_diameter, wall_thickness, corrosion_rate, pipeline_age)
failure_probability = calculate_failure_probability(internal_pressure, failure_pressure, corrosion_rate, pipeline_age)

with col1:
    st.metric("Failure Pressure Estimation (Barlow)", f"{failure_pressure:,.2f} psi")
with col2:
    st.metric("Calculated Likelihood of Structural Failure", f"{failure_probability*100:.2f}%")

# Create DataFrame explicitly structured for inference predictability
input_data_dict = {
    'Reservoir_Pressure_psi': [res_pressure],
    'Reservoir_Temperature_C': [res_temp],
    'Oil_Production_Rate_bbl_day': [oil_rate],
    'Gas_Production_Rate_MSCF_day': [gas_rate],
    'Water_Cut_percent': [water_cut],
    'Pipeline_Diameter_m': [pipeline_diameter],
    'Wall_Thickness_mm': [wall_thickness],
    'Pipeline_Length_km': [pipeline_length],
    'Flow_Velocity_m_s': [flow_velocity],
    'Fluid_Density_kg_m3': [fluid_density],
    'Fluid_Viscosity_cP': [fluid_viscosity],
    'Corrosion_Rate_mm_year': [corrosion_rate],
    'Internal_Pressure_psi': [internal_pressure],
    'Temperature_Gradient_C_km': [temp_gradient],
    'Elevation_Change_m': [elevation_change],
    'Pipeline_Age_years': [pipeline_age],
    'Calculated_Failure_Pressure_psi': [failure_pressure]
}
input_data = pd.DataFrame(input_data_dict)
input_data = input_data[feature_names]

# Machine Learning Classification Action
st.header("2. AI Risk Diagnosis System")
if st.button("Query RSPM Neural/ML Risk Model"):
    # Pre-process Data Vector via fitted scaler
    scaled_input = scaler.transform(input_data)
    prediction_class = model.predict(scaled_input)[0]
    
    # Assess Contextual Severity status
    health_status = determine_pipeline_health_status(prediction_class, failure_probability)
    
    # Visual Alert Triage Trigger
    if prediction_class == 'High':
        st.error(f"🔴 AI Classification: {prediction_class} Risk | Operational Status: {health_status}")
    elif prediction_class == 'Medium':
        st.warning(f"🟡 AI Classification: {prediction_class} Risk | Operational Status: {health_status}")
    else:
        st.success(f"🟢 AI Classification: {prediction_class} Risk  | Operational Status: {health_status}")
        
    st.markdown("### Export Capabilities")
    
    # Ready payload configurations for Reports
    export_payload = input_data.iloc[0].to_dict()
    export_payload['Predicted_Leak_Risk_Class'] = prediction_class
    export_payload['Failure_Probability_Percent'] = f"{failure_probability*100:.2f}%"
    export_payload['Infrastructure_Health_Status'] = health_status
    
    col_a, col_b = st.columns(2)
    with col_a:
        excel_path = generate_excel_report(export_payload)
        with open(excel_path, "rb") as file_excel:
            st.download_button("Download Export (Excel)", file_excel, file_name="RSPM_Risk_Assessment.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col_b:
        pdf_path = generate_pdf_report(export_payload)
        with open(pdf_path, "rb") as file_pdf:
            st.download_button("Download Report (PDF)", file_pdf, file_name="RSPM_Risk_Assessment.pdf", mime="application/pdf")

# Dashboard Charts & Distributions Simulation
st.header("3. Pipeline Degradation & Risk Projections")
st.markdown("Forecasting failure probability driven by long-term corrosion degradation trajectory.")

ages_range = np.arange(1, 41)
simulated_probabilities = []
for test_age in ages_range:
    test_p_fail = calculate_failure_pressure(pipeline_diameter, wall_thickness, corrosion_rate, test_age)
    prob = calculate_failure_probability(internal_pressure, test_p_fail, corrosion_rate, test_age)
    simulated_probabilities.append(prob)

fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(x=ages_range, y=simulated_probabilities, color='orangered', ax=ax, linewidth=2)
ax.axhline(0.35, color='gold', linestyle='--', label='Warning Threshold (0.35)')
ax.axhline(0.70, color='red', linestyle='--', label='Critical Threshold (0.70)')
ax.set_title("Probability of Failure Progression Against Asset Lifespan")
ax.set_xlabel("Operational Asset Age (Years)")
ax.set_ylabel("Likelihood of Failure")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
