import numpy as np
import pandas as pd

from scripts.pipeline_model import calculate_failure_pressure


RAW_INPUT_FEATURES = [
    "Reservoir_Pressure_psi",
    "Reservoir_Temperature_C",
    "Oil_Production_Rate_bbl_day",
    "Gas_Production_Rate_MSCF_day",
    "Water_Cut_percent",
    "Pipeline_Diameter_m",
    "Wall_Thickness_mm",
    "Pipeline_Length_km",
    "Flow_Velocity_m_s",
    "Fluid_Density_kg_m3",
    "Fluid_Viscosity_cP",
    "Corrosion_Rate_mm_year",
    "Internal_Pressure_psi",
    "Temperature_Gradient_C_km",
    "Elevation_Change_m",
    "Pipeline_Age_years",
]

OPTIONAL_INPUT_FEATURES = ["Calculated_Failure_Pressure_psi"]

RESERVOIR_FEATURES = [
    "Reservoir_Pressure_psi",
    "Reservoir_Temperature_C",
    "Oil_Production_Rate_bbl_day",
    "Gas_Production_Rate_MSCF_day",
    "Water_Cut_percent",
    "Gas_Oil_Ratio",
    "Water_Oil_Ratio",
    "Pressure_Drawdown_psi",
    "Thermal_Load_Index",
    "Production_Utilization_Index",
]

PIPELINE_FEATURES = [
    "Pipeline_Diameter_m",
    "Wall_Thickness_mm",
    "Flow_Velocity_m_s",
    "Fluid_Density_kg_m3",
    "Corrosion_Rate_mm_year",
    "Internal_Pressure_psi",
    "Temperature_Gradient_C_km",
    "Pipeline_Age_years",
    "Calculated_Failure_Pressure_psi",
    "Pressure_to_Failure_Ratio",
    "Integrity_Margin_psi",
    "Hydraulic_Stress_Index",
]

INTEGRATED_FEATURES = RESERVOIR_FEATURES + PIPELINE_FEATURES

RISK_ORDER = ["No Leak", "Minor Leak", "Moderate Leak", "Major Leak"]

DECISION_ACTIONS = {
    "No Leak": "Monitor",
    "Minor Leak": "Inspect",
    "Moderate Leak": "Intervene",
    "Major Leak": "Shut In Well",
}


def _clip_series(series, lower=0.0):
    return np.clip(series.astype(float), lower, None)


def _normalize(series, min_value, max_value):
    scaled = (series.astype(float) - min_value) / (max_value - min_value)
    return np.clip(scaled, 0.0, 1.0)


def ensure_base_input_frame(df):
    missing = [column for column in RAW_INPUT_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required base input columns: {', '.join(missing)}")

    prepared = df.copy()
    if "Calculated_Failure_Pressure_psi" not in prepared.columns:
        prepared["Calculated_Failure_Pressure_psi"] = prepared.apply(
            lambda row: calculate_failure_pressure(
                row["Pipeline_Diameter_m"],
                row["Wall_Thickness_mm"],
                row["Corrosion_Rate_mm_year"],
                row["Pipeline_Age_years"],
            ),
            axis=1,
        )
    return prepared


def build_integrated_feature_frame(df):
    base = ensure_base_input_frame(df)

    oil_rate = _clip_series(base["Oil_Production_Rate_bbl_day"], lower=1.0)
    gas_rate = _clip_series(base["Gas_Production_Rate_MSCF_day"], lower=0.0)
    water_cut = np.clip(base["Water_Cut_percent"].astype(float), 0.0, 100.0)
    internal_pressure = base["Internal_Pressure_psi"].astype(float)
    reservoir_pressure = base["Reservoir_Pressure_psi"].astype(float)
    failure_pressure = _clip_series(base["Calculated_Failure_Pressure_psi"], lower=1.0)
    flow_velocity = _clip_series(base["Flow_Velocity_m_s"], lower=0.0)
    fluid_density = _clip_series(base["Fluid_Density_kg_m3"], lower=1.0)

    gas_oil_ratio = gas_rate / oil_rate
    water_oil_ratio = water_cut / np.clip(100.0 - water_cut, 1.0, None)
    pressure_drawdown = np.clip(reservoir_pressure - internal_pressure, 0.0, None)
    thermal_load = (
        base["Reservoir_Temperature_C"].astype(float)
        * base["Temperature_Gradient_C_km"].astype(float)
    )
    production_utilization = (
        oil_rate / 50000.0
        + gas_rate / 20000.0
        + water_cut / 100.0
    ) / 3.0
    pressure_ratio = internal_pressure / failure_pressure
    integrity_margin = np.clip(failure_pressure - internal_pressure, 0.0, None)
    hydraulic_stress = flow_velocity * fluid_density * pressure_ratio

    feature_frame = pd.DataFrame(
        {
            "Reservoir_Pressure_psi": reservoir_pressure,
            "Reservoir_Temperature_C": base["Reservoir_Temperature_C"].astype(float),
            "Oil_Production_Rate_bbl_day": oil_rate,
            "Gas_Production_Rate_MSCF_day": gas_rate,
            "Water_Cut_percent": water_cut,
            "Gas_Oil_Ratio": gas_oil_ratio,
            "Water_Oil_Ratio": water_oil_ratio,
            "Pressure_Drawdown_psi": pressure_drawdown,
            "Thermal_Load_Index": thermal_load,
            "Production_Utilization_Index": production_utilization,
            "Pipeline_Diameter_m": base["Pipeline_Diameter_m"].astype(float),
            "Wall_Thickness_mm": base["Wall_Thickness_mm"].astype(float),
            "Flow_Velocity_m_s": flow_velocity,
            "Fluid_Density_kg_m3": fluid_density,
            "Corrosion_Rate_mm_year": base["Corrosion_Rate_mm_year"].astype(float),
            "Internal_Pressure_psi": internal_pressure,
            "Temperature_Gradient_C_km": base["Temperature_Gradient_C_km"].astype(float),
            "Pipeline_Age_years": base["Pipeline_Age_years"].astype(float),
            "Calculated_Failure_Pressure_psi": failure_pressure,
            "Pressure_to_Failure_Ratio": pressure_ratio,
            "Integrity_Margin_psi": integrity_margin,
            "Hydraulic_Stress_Index": hydraulic_stress,
        }
    )

    return feature_frame[INTEGRATED_FEATURES]


def compute_lrs_components(df):
    base = ensure_base_input_frame(df)
    features = build_integrated_feature_frame(base)

    frd = _normalize(features["Pressure_to_Failure_Ratio"], 0.35, 1.10) * 100.0
    corrosion = _normalize(features["Corrosion_Rate_mm_year"], 0.01, 2.0) * 100.0
    pda = _normalize(features["Pressure_Drawdown_psi"], 500.0, 5000.0) * 100.0
    water_cut = _normalize(features["Water_Cut_percent"], 0.0, 90.0) * 100.0
    gor = _normalize(features["Gas_Oil_Ratio"], 0.0, 8.0) * 100.0
    utilization = np.clip(features["Production_Utilization_Index"], 0.0, 1.0) * 100.0

    return pd.DataFrame(
        {
            "FRD": frd,
            "CR": corrosion,
            "PDA": pda,
            "WC": water_cut,
            "GOR": gor,
            "PU": utilization,
        }
    )


def compute_lrs_score(df):
    components = compute_lrs_components(df)
    return (
        0.25 * components["FRD"]
        + 0.25 * components["CR"]
        + 0.20 * components["PDA"]
        + 0.15 * components["WC"]
        + 0.10 * components["GOR"]
        + 0.05 * components["PU"]
    )


def lrs_to_risk_class(lrs_score):
    if lrs_score < 20:
        return "No Leak"
    if lrs_score < 50:
        return "Minor Leak"
    if lrs_score < 75:
        return "Moderate Leak"
    return "Major Leak"


def resolve_decision_output(class_prediction, lrs_score, corrosion_rate, pressure_ratio):
    severity_map = {label: index for index, label in enumerate(RISK_ORDER)}
    class_index = severity_map.get(str(class_prediction), 0)
    lrs_index = severity_map[lrs_to_risk_class(float(lrs_score))]

    escalated_index = max(class_index, lrs_index)
    if float(corrosion_rate) >= 1.5 or float(pressure_ratio) >= 0.95:
        escalated_index = max(escalated_index, 3)
    elif float(corrosion_rate) >= 0.9 or float(pressure_ratio) >= 0.75:
        escalated_index = max(escalated_index, 2)

    decision_label = RISK_ORDER[escalated_index]
    return {
        "decision_label": decision_label,
        "decision_action": DECISION_ACTIONS[decision_label],
    }


def augment_with_architecture(df):
    base = ensure_base_input_frame(df)
    features = build_integrated_feature_frame(base)
    components = compute_lrs_components(base)
    lrs_scores = compute_lrs_score(base)
    architecture_class = lrs_scores.apply(lrs_to_risk_class)

    augmented = base.copy()
    for column in INTEGRATED_FEATURES:
        augmented[column] = features[column]
    for column in components.columns:
        augmented[f"LRS_{column}"] = components[column]
    augmented["Architecture_LRS"] = lrs_scores
    augmented["Architecture_Risk_Class"] = architecture_class
    augmented["Decision_Action"] = architecture_class.map(DECISION_ACTIONS)
    return augmented


def build_single_scenario_frame(values):
    return pd.DataFrame(
        [
            {
                "Reservoir_Pressure_psi": values["res_pressure"],
                "Reservoir_Temperature_C": values["res_temp"],
                "Oil_Production_Rate_bbl_day": values["oil_rate"],
                "Gas_Production_Rate_MSCF_day": values["gas_rate"],
                "Water_Cut_percent": values["water_cut"],
                "Pipeline_Diameter_m": values["pipeline_diameter"],
                "Wall_Thickness_mm": values["wall_thickness"],
                "Pipeline_Length_km": values["pipeline_length"],
                "Flow_Velocity_m_s": values["flow_velocity"],
                "Fluid_Density_kg_m3": values["fluid_density"],
                "Fluid_Viscosity_cP": values["fluid_viscosity"],
                "Corrosion_Rate_mm_year": values["corrosion_rate"],
                "Internal_Pressure_psi": values["internal_pressure"],
                "Temperature_Gradient_C_km": values["temp_gradient"],
                "Elevation_Change_m": values["elevation_change"],
                "Pipeline_Age_years": values["pipeline_age"],
            }
        ]
    )


def summarize_architecture(values):
    base_frame = build_single_scenario_frame(values)
    augmented = augment_with_architecture(base_frame)
    row = augmented.iloc[0]
    decision = resolve_decision_output(
        row["Architecture_Risk_Class"],
        row["Architecture_LRS"],
        row["Corrosion_Rate_mm_year"],
        row["Pressure_to_Failure_Ratio"],
    )

    return {
        "base_frame": base_frame,
        "integrated_frame": augmented[INTEGRATED_FEATURES],
        "lrs_score": float(row["Architecture_LRS"]),
        "risk_class": row["Architecture_Risk_Class"],
        "decision_label": decision["decision_label"],
        "decision_action": decision["decision_action"],
        "pressure_ratio": float(row["Pressure_to_Failure_Ratio"]),
        "integrity_margin": float(row["Integrity_Margin_psi"]),
        "failure_pressure": float(row["Calculated_Failure_Pressure_psi"]),
        "components": {
            "FRD": float(row["LRS_FRD"]),
            "CR": float(row["LRS_CR"]),
            "PDA": float(row["LRS_PDA"]),
            "WC": float(row["LRS_WC"]),
            "GOR": float(row["LRS_GOR"]),
            "PU": float(row["LRS_PU"]),
        },
    }
