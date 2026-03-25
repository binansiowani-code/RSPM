import numpy as np

def calculate_failure_pressure(diameter_m, wall_thickness_mm, corrosion_rate_mm_year, age_years=10, yield_strength_psi=60000):
    """
    Calculate the failure pressure using Barlow's equation.
    Formula: P = 2 * S * t / D
    """
    # Effective thickness calculation
    effective_thickness_mm = max(wall_thickness_mm - (corrosion_rate_mm_year * age_years), 1.0)
    # Barlow's equation calculation
    failure_pressure_psi = (2 * yield_strength_psi * effective_thickness_mm) / (diameter_m * 1000.0)
    return failure_pressure_psi

def calculate_failure_probability(internal_pressure_psi, failure_pressure_psi, corrosion_rate_mm_year, age_years=10):
    """
    Estimate failure probability based on internal pressure, failure pressure, and corrosion progression.
    """
    pressure_ratio = internal_pressure_psi / failure_pressure_psi
    
    # Calculate base failure probability using a sigmoid mapping of the ratio
    base_prob = 1.0 / (1.0 + np.exp(-10 * (pressure_ratio - 0.5)))
    
    # Adding linear factors for age and corrosion directly
    corrosion_factor = corrosion_rate_mm_year / 2.0
    age_factor = age_years / 40.0
    
    failure_probability = base_prob * 0.6 + corrosion_factor * 0.2 + age_factor * 0.2
    
    return float(np.clip(failure_probability, 0.0, 1.0))

def determine_pipeline_health_status(leak_risk_class, failure_probability):
    """
    A logical categorization for comprehensive pipeline health status
    """
    if leak_risk_class == 'High' or failure_probability >= 0.70:
        return "Critical - Immediate Maintenance Required"
    elif leak_risk_class == 'Medium' or failure_probability >= 0.35:
        return "Warning - Moderate Degradation, Schedule Inspection"
    else:
        return "Healthy - Normal Operation"
