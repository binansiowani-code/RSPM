import pandas as pd
import numpy as np
import os

def generate_dataset(num_samples=1500, output_path=None):
    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'rspm_integrated.csv'))
    np.random.seed(42)
    
    # Reservoir Parameters
    res_pressure = np.random.uniform(3000, 8000, num_samples)
    res_temp = np.random.uniform(60, 150, num_samples)
    oil_rate = np.random.uniform(1000, 50000, num_samples)
    gas_rate = np.random.uniform(500, 20000, num_samples)
    water_cut = np.random.uniform(0, 90, num_samples)
    
    # Pipeline Physical Parameters
    diameter_m = np.random.uniform(0.1, 1.0, num_samples)
    wall_thickness_mm = np.random.uniform(5, 25, num_samples)
    length_km = np.random.uniform(5, 100, num_samples)
    flow_velocity = np.random.uniform(0.5, 5.0, num_samples)
    fluid_density = np.random.uniform(700, 1000, num_samples)
    fluid_viscosity = np.random.uniform(1, 50, num_samples)
    
    # Operational Conditions
    corrosion_rate = np.random.uniform(0.01, 2.0, num_samples)
    internal_pressure = res_pressure * np.random.uniform(0.3, 0.8, num_samples) # Internal pressure usually drops from reservoir
    temp_gradient = np.random.uniform(0.5, 5.0, num_samples)
    elevation_change = np.random.uniform(-200, 200, num_samples)
    pipeline_age_years = np.random.uniform(1, 40, num_samples)
    
    # Engineering Calculations to determine Target Variables
    # Yield strength of standard pipeline steel (X60) is ~ 60,000 psi
    yield_strength = 60000 
    
    # Effective thickness = original thickness - corrosion over operational life
    effective_thickness_mm = np.maximum(wall_thickness_mm - (corrosion_rate * pipeline_age_years), 1.0)
    
    # Barlow's equation: Failure Pressure (psi) = 2 * Yield Strength * Thickness / Diameter
    failure_pressure = (2 * yield_strength * effective_thickness_mm) / (diameter_m * 1000)
    
    # Failure Probability calculation based on ratio of internal pressure to failure pressure and corrosion progression
    pressure_ratio = internal_pressure / failure_pressure
    
    # Generate failure probability utilizing sigmoid to constrain between 0 and 1, combined with noise
    base_prob = 1.0 / (1.0 + np.exp(-10 * (pressure_ratio - 0.5)))
    corrosion_factor = corrosion_rate / 2.0
    age_factor = pipeline_age_years / 40.0
    
    failure_probability = base_prob * 0.6 + corrosion_factor * 0.2 + age_factor * 0.2
    failure_probability = np.clip(failure_probability + np.random.normal(0, 0.05, num_samples), 0, 1)
    
    # Target Classes Assignment
    conditions = [
        failure_probability < 0.35,
        (failure_probability >= 0.35) & (failure_probability < 0.70),
        failure_probability >= 0.70
    ]
    choices = ['Low', 'Medium', 'High']
    leak_risk_class = np.select(conditions, choices, default='Unknown')
    
    df = pd.DataFrame({
        'Reservoir_Pressure_psi': res_pressure,
        'Reservoir_Temperature_C': res_temp,
        'Oil_Production_Rate_bbl_day': oil_rate,
        'Gas_Production_Rate_MSCF_day': gas_rate,
        'Water_Cut_percent': water_cut,
        'Pipeline_Diameter_m': diameter_m,
        'Wall_Thickness_mm': wall_thickness_mm,
        'Pipeline_Length_km': length_km,
        'Flow_Velocity_m_s': flow_velocity,
        'Fluid_Density_kg_m3': fluid_density,
        'Fluid_Viscosity_cP': fluid_viscosity,
        'Corrosion_Rate_mm_year': corrosion_rate,
        'Internal_Pressure_psi': internal_pressure,
        'Temperature_Gradient_C_km': temp_gradient,
        'Elevation_Change_m': elevation_change,
        'Pipeline_Age_years': pipeline_age_years,
        'Calculated_Failure_Pressure_psi': failure_pressure,
        'Failure_Probability': failure_probability,
        'Leak_Risk_Class': leak_risk_class
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully generated at {output_path} with {num_samples} samples.")

if __name__ == "__main__":
    generate_dataset()
