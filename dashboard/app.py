import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.pipeline_model import (  # noqa: E402
    calculate_failure_pressure,
    calculate_failure_probability,
    determine_pipeline_health_status,
)
from scripts.architecture_engine import (  # noqa: E402
    DECISION_ACTIONS,
    INTEGRATED_FEATURES,
    RAW_INPUT_FEATURES,
    augment_with_architecture,
    build_integrated_feature_frame,
    build_single_scenario_frame,
    resolve_decision_output,
    summarize_architecture,
)
from scripts.report_generator import generate_excel_report, generate_pdf_report  # noqa: E402


RISK_COLORS = {
    0: "#27AE60",
    1: "#F39C12",
    2: "#E67E22",
    3: "#CE1A19",
}

CATEGORY_COLORS = {
    "Reservoir": "#4cc9f0",
    "Pipeline": "#f7b267",
    "Fluid Chemistry": "#4fd1a5",
}


st.set_page_config(
    page_title="RSPM Command Center",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_ml_assets():
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "rspm_models.pkl")
    )
    return joblib.load(model_path)


@st.cache_data
def load_integrated_dataset():
    dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "rspm_integrated.csv")
    )
    return augment_with_architecture(pd.read_csv(dataset_path))


@st.cache_data
def load_analysis_summary():
    analysis_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "data", "RSPM_Integrated_Analysis.csv"
        )
    )
    if not os.path.exists(analysis_path):
        return None
    return pd.read_csv(analysis_path, header=None).fillna("")


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg: #07111f;
            --panel: rgba(10, 24, 42, 0.78);
            --panel-strong: rgba(11, 30, 52, 0.95);
            --border: rgba(126, 169, 255, 0.18);
            --text: #eaf3ff;
            --muted: #8ea6c6;
            --accent: #4cc9f0;
            --accent-2: #f7b267;
            --success: #4fd1a5;
            --warning: #ffd166;
            --danger: #ff6b6b;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(76, 201, 240, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(247, 178, 103, 0.14), transparent 24%),
                linear-gradient(180deg, #040a14 0%, #07111f 45%, #091729 100%);
            color: var(--text);
            font-family: "IBM Plex Sans", sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7, 17, 31, 0.98), rgba(9, 23, 41, 0.98));
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
            font-family: "IBM Plex Sans", sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
            color: var(--text);
        }

        .hero {
            padding: 2rem 2rem 1.6rem 2rem;
            border: 1px solid var(--border);
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(16, 38, 67, 0.92), rgba(10, 24, 42, 0.92)),
                linear-gradient(135deg, rgba(76, 201, 240, 0.12), rgba(247, 178, 103, 0.08));
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.24);
            overflow: hidden;
            position: relative;
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: auto -80px -120px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(76, 201, 240, 0.22), transparent 70%);
        }

        .eyebrow {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(76, 201, 240, 0.12);
            border: 1px solid rgba(76, 201, 240, 0.28);
            color: #b9ebff;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .hero h1 {
            margin: 0.9rem 0 0.7rem 0;
            font-size: 3rem;
            line-height: 1.02;
        }

        .hero p {
            margin: 0;
            max-width: 760px;
            font-size: 1.02rem;
            color: var(--muted);
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.5rem;
        }

        .mini-tile, .stat-card, .glass-card {
            border: 1px solid var(--border);
            border-radius: 22px;
            background: var(--panel);
            backdrop-filter: blur(10px);
        }

        .mini-tile {
            padding: 1rem 1.1rem;
        }

        .mini-tile-label,
        .stat-label,
        .section-kicker {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .mini-tile-value {
            margin-top: 0.35rem;
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text);
        }

        .section-header {
            margin-top: 1.4rem;
            margin-bottom: 0.8rem;
        }

        .section-header h2 {
            margin-bottom: 0.15rem;
        }

        .section-header p {
            margin: 0;
            color: var(--muted);
        }

        .stat-card {
            padding: 1.15rem 1.2rem 1rem 1.2rem;
            min-height: 150px;
        }

        .stat-value {
            margin-top: 0.4rem;
            font-family: "Space Grotesk", sans-serif;
            font-size: 2rem;
            font-weight: 700;
        }

        .stat-footnote {
            margin-top: 0.7rem;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .glass-card {
            padding: 1.25rem;
        }

        .risk-banner {
            padding: 1.2rem 1.3rem;
            border-radius: 22px;
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .risk-banner h3 {
            margin: 0 0 0.35rem 0;
            color: #fff;
        }

        .risk-banner p {
            margin: 0;
            color: rgba(255, 255, 255, 0.88);
        }

        .risk-high {
            background: linear-gradient(135deg, rgba(148, 29, 39, 0.98), rgba(255, 107, 107, 0.85));
        }

        .risk-medium {
            background: linear-gradient(135deg, rgba(132, 92, 12, 0.98), rgba(255, 209, 102, 0.78));
        }

        .risk-low {
            background: linear-gradient(135deg, rgba(18, 93, 75, 0.98), rgba(79, 209, 165, 0.78));
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            margin-top: 0.8rem;
            background: rgba(255, 255, 255, 0.08);
            color: #fff;
            font-size: 0.9rem;
        }

        .table-frame {
            border: 1px solid var(--border);
            border-radius: 22px;
            overflow: hidden;
            background: var(--panel);
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(76, 201, 240, 0.45);
            background: linear-gradient(135deg, rgba(76, 201, 240, 0.92), rgba(53, 184, 213, 0.88));
            color: #03111d;
            font-weight: 700;
            padding: 0.7rem 1.2rem;
        }

        .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.06);
            color: var(--text);
            font-weight: 600;
        }

        div[data-testid="stMetric"] {
            background: transparent;
            border: none;
            padding: 0;
        }

        @media (max-width: 900px) {
            .hero h1 {
                font-size: 2.35rem;
            }

            .hero-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title, description):
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-kicker">Operational View</div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label, value, footnote, accent):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value" style="color:{accent};">{value}</div>
            <div class="stat-footnote">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_health_style(health_status):
    if "Critical" in health_status:
        return "risk-high", RISK_COLORS[3]
    if "Warning" in health_status:
        return "risk-medium", RISK_COLORS[1]
    return "risk-low", RISK_COLORS[0]


def get_risk_color_by_probability(probability):
    if probability >= 0.70:
        return RISK_COLORS[3]
    if probability >= 0.50:
        return RISK_COLORS[2]
    if probability >= 0.35:
        return RISK_COLORS[1]
    return RISK_COLORS[0]


def get_class_color(label):
    label_text = str(label).strip().lower()
    mapping = {
        "0": RISK_COLORS[0],
        "no leak": RISK_COLORS[0],
        "low": RISK_COLORS[0],
        "1": RISK_COLORS[1],
        "minor": RISK_COLORS[1],
        "minor leak": RISK_COLORS[1],
        "medium": RISK_COLORS[1],
        "2": RISK_COLORS[2],
        "moderate": RISK_COLORS[2],
        "moderate leak": RISK_COLORS[2],
        "3": RISK_COLORS[3],
        "major": RISK_COLORS[3],
        "major leak": RISK_COLORS[3],
        "high": RISK_COLORS[3],
    }
    return mapping.get(label_text, "#8ea6c6")


def build_input_frame(values):
    return build_single_scenario_frame(values)


def validate_uploaded_dataset(df, raw_input_features):
    missing_required = [name for name in raw_input_features if name not in df.columns]
    if missing_required:
        return False, missing_required
    return True, []


def looks_like_summary_csv(df):
    flattened = " ".join(df.astype(str).fillna("").stack().tolist()).lower()
    summary_markers = [
        "integrated model performance summary",
        "class distribution",
        "final results",
        "model",
        "test accuracy",
    ]
    return any(marker in flattened for marker in summary_markers)


def build_upload_template(raw_input_features):
    template_row = {
        "Reservoir_Pressure_psi": 4500.0,
        "Reservoir_Temperature_C": 85.0,
        "Oil_Production_Rate_bbl_day": 15000.0,
        "Gas_Production_Rate_MSCF_day": 8000.0,
        "Water_Cut_percent": 30.0,
        "Pipeline_Diameter_m": 0.4,
        "Wall_Thickness_mm": 10.0,
        "Pipeline_Length_km": 35.0,
        "Flow_Velocity_m_s": 2.5,
        "Fluid_Density_kg_m3": 850.0,
        "Fluid_Viscosity_cP": 12.0,
        "Corrosion_Rate_mm_year": 0.25,
        "Internal_Pressure_psi": 1200.0,
        "Temperature_Gradient_C_km": 1.5,
        "Elevation_Change_m": -10.0,
        "Pipeline_Age_years": 15.0,
    }
    return pd.DataFrame([[template_row[name] for name in raw_input_features]], columns=raw_input_features)


def build_parameter_categories(values, failure_pressure):
    flow_rate = values["oil_rate"] + values["gas_rate"]
    categories = pd.DataFrame(
        [
            {
                "Category": "Reservoir",
                "Parameter": "Temperature",
                "Value": values["res_temp"],
                "Unit": "deg C",
                "Status": "Measured",
            },
            {
                "Category": "Reservoir",
                "Parameter": "Flow Rate",
                "Value": flow_rate,
                "Unit": "combined oil+gas",
                "Status": "Derived",
            },
            {
                "Category": "Pipeline",
                "Parameter": "Pipeline Diameter",
                "Value": values["pipeline_diameter"],
                "Unit": "m",
                "Status": "Measured",
            },
            {
                "Category": "Pipeline",
                "Parameter": "Wall Thickness",
                "Value": values["wall_thickness"],
                "Unit": "mm",
                "Status": "Measured",
            },
            {
                "Category": "Pipeline",
                "Parameter": "Corrosion Rate",
                "Value": values["corrosion_rate"],
                "Unit": "mm/year",
                "Status": "Measured",
            },
            {
                "Category": "Pipeline",
                "Parameter": "Failure Pressure",
                "Value": failure_pressure,
                "Unit": "psi",
                "Status": "Calculated",
            },
            {
                "Category": "Fluid Chemistry",
                "Parameter": "H2O Content",
                "Value": values["water_cut"],
                "Unit": "% water cut",
                "Status": "Proxy",
            },
            {
                "Category": "Fluid Chemistry",
                "Parameter": "CO2 Content",
                "Value": np.nan,
                "Unit": "not available",
                "Status": "Unavailable",
            },
        ]
    )
    return categories


def make_display_name(sample_row, sample_index):
    risk_label = sample_row.get(
        "Architecture_Risk_Class",
        sample_row.get("Leak_Risk_Class", "Scenario"),
    )
    lrs_score = float(sample_row.get("Architecture_LRS", 0.0))
    return (
        f"Sample {sample_index:04d} | "
        f"{risk_label} | "
        f"LRS {lrs_score:.1f}"
    )


def normalize(value, min_value, max_value):
    return float(np.clip((value - min_value) / (max_value - min_value), 0.0, 1.0))


def make_risk_profile(values, failure_probability, pressure_ratio):
    profile = {
        "Pressure load": max(pressure_ratio, 0.0),
        "Corrosion": normalize(values["corrosion_rate"], 0.01, 2.0),
        "Asset age": normalize(values["pipeline_age"], 1.0, 40.0),
        "Water cut": normalize(values["water_cut"], 0.0, 90.0),
        "Flow velocity": normalize(values["flow_velocity"], 0.5, 5.0),
        "Thermal gradient": normalize(values["temp_gradient"], 0.5, 5.0),
        "Overall failure risk": failure_probability,
    }
    return pd.Series(profile).sort_values(ascending=True)


def generate_projection_chart(values):
    ages_range = np.arange(1, 41)
    probabilities = []

    for test_age in ages_range:
        failure_pressure = calculate_failure_pressure(
            values["pipeline_diameter"],
            values["wall_thickness"],
            values["corrosion_rate"],
            test_age,
        )
        probability = calculate_failure_probability(
            values["internal_pressure"],
            failure_pressure,
            values["corrosion_rate"],
            test_age,
        )
        probabilities.append(probability)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor("#0b1a2c")
    ax.set_facecolor("#0f2238")

    probability_array = np.array(probabilities)
    low_mask = probability_array < 0.35
    mid_mask = (probability_array >= 0.35) & (probability_array < 0.70)
    high_mask = probability_array >= 0.70

    ax.fill_between(ages_range, 0, probability_array, where=low_mask, color=RISK_COLORS[0], alpha=0.20)
    ax.fill_between(ages_range, 0, probability_array, where=mid_mask, color=RISK_COLORS[1], alpha=0.20)
    ax.fill_between(ages_range, 0, probability_array, where=high_mask, color=RISK_COLORS[3], alpha=0.24)
    ax.plot(ages_range, probabilities, color="#dcecff", linewidth=2.6, solid_capstyle="round")
    ax.scatter(
        ages_range[-1],
        probabilities[-1],
        s=130,
        color=get_risk_color_by_probability(probabilities[-1]),
        edgecolor="#f7fbff",
        linewidth=1.5,
        zorder=5,
    )
    ax.axhline(0.35, color=RISK_COLORS[1], linestyle="--", linewidth=1.4)
    ax.axhline(0.70, color=RISK_COLORS[3], linestyle="--", linewidth=1.4)

    ax.set_title("Failure Probability Through Asset Life", color="#eaf3ff", fontsize=15, pad=14)
    ax.set_xlabel("Operational age (years)", color="#9fb6d3")
    ax.set_ylabel("Probability of failure", color="#9fb6d3")
    ax.tick_params(colors="#b9cbe2")
    ax.grid(color="#36506e", alpha=0.28)
    for spine in ax.spines.values():
        spine.set_color("#274261")

    return fig


def generate_driver_chart(risk_profile):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor("#0b1a2c")
    ax.set_facecolor("#0f2238")

    colors = [get_risk_color_by_probability(value) for value in risk_profile.values]
    positions = np.arange(len(risk_profile))
    ax.hlines(y=positions, xmin=0, xmax=risk_profile.values, color=colors, linewidth=7, alpha=0.95)
    ax.scatter(risk_profile.values, positions, s=120, color=colors, edgecolor="#eff7ff", linewidth=1.2, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_yticks(positions)
    ax.set_yticklabels(risk_profile.index)
    ax.set_title("Current Risk Drivers", color="#eaf3ff", fontsize=15, pad=14)
    ax.set_xlabel("Normalized intensity", color="#9fb6d3")
    ax.tick_params(colors="#b9cbe2")
    ax.grid(axis="x", color="#36506e", alpha=0.28)
    for spine in ax.spines.values():
        spine.set_color("#274261")

    return fig


def generate_category_chart(category_df):
    plot_df = category_df.dropna(subset=["Value"]).copy()
    value_strings = []
    for _, row in plot_df.iterrows():
        if row["Parameter"] == "Flow Rate":
            value_strings.append(f"{row['Value']:,.0f}")
        elif row["Parameter"] in {"Pipeline Diameter", "Corrosion Rate"}:
            value_strings.append(f"{row['Value']:.2f}")
        elif row["Parameter"] == "Wall Thickness":
            value_strings.append(f"{row['Value']:.1f}")
        elif row["Parameter"] == "Temperature":
            value_strings.append(f"{row['Value']:.1f}")
        else:
            value_strings.append(f"{row['Value']:,.0f}")
    plot_df["Display"] = value_strings

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor("#0b1a2c")
    ax.set_facecolor("#0f2238")

    colors = [CATEGORY_COLORS.get(category, "#b9cbe2") for category in plot_df["Category"]]
    positions = np.arange(len(plot_df))
    bars = ax.bar(positions, plot_df["Value"], color=colors, width=0.58)

    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df["Parameter"], rotation=20, ha="right", color="#b9cbe2")
    ax.tick_params(axis="y", colors="#b9cbe2")
    ax.set_title("Categorised Parameter Overview", color="#eaf3ff", fontsize=15, pad=14)
    ax.grid(axis="y", color="#36506e", alpha=0.28)
    for spine in ax.spines.values():
        spine.set_color("#274261")

    for bar, value in zip(bars, plot_df["Display"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            value,
            ha="center",
            va="bottom",
            color="#eaf3ff",
            fontsize=8,
        )

    return fig


def generate_pipeline_reservoir_comparison(values, failure_pressure):
    comparison_df = pd.DataFrame(
        [
            {"Group": "Reservoir", "Metric": "Temperature", "Value": values["res_temp"]},
            {"Group": "Reservoir", "Metric": "Flow Rate", "Value": values["oil_rate"] + values["gas_rate"]},
            {"Group": "Pipeline", "Metric": "Diameter", "Value": values["pipeline_diameter"]},
            {"Group": "Pipeline", "Metric": "Wall Thickness", "Value": values["wall_thickness"]},
            {"Group": "Pipeline", "Metric": "Corrosion Rate", "Value": values["corrosion_rate"]},
            {"Group": "Pipeline", "Metric": "Failure Pressure", "Value": failure_pressure},
        ]
    )
    scaling_map = {
        "Temperature": 150.0,
        "Flow Rate": 70000.0,
        "Diameter": 1.0,
        "Wall Thickness": 25.0,
        "Corrosion Rate": 2.0,
        "Failure Pressure": 30000.0,
    }
    comparison_df["Normalized"] = comparison_df.apply(
        lambda row: min(row["Value"] / scaling_map[row["Metric"]], 1.0),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.patch.set_facecolor("#0b1a2c")
    ax.set_facecolor("#0f2238")

    sns.violinplot(
        data=comparison_df,
        x="Metric",
        y="Normalized",
        hue="Group",
        palette={"Reservoir": CATEGORY_COLORS["Reservoir"], "Pipeline": CATEGORY_COLORS["Pipeline"]},
        inner=None,
        cut=0,
        ax=ax,
    )
    sns.stripplot(
        data=comparison_df,
        x="Metric",
        y="Normalized",
        hue="Group",
        dodge=True,
        palette={"Reservoir": "#d8f5ff", "Pipeline": "#ffe2bb"},
        size=8,
        edgecolor="#0f2238",
        linewidth=0.7,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Reservoir vs Pipeline Emphasis", color="#eaf3ff", fontsize=15, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Relative scale", color="#9fb6d3")
    ax.tick_params(axis="x", colors="#b9cbe2", rotation=20)
    ax.tick_params(axis="y", colors="#b9cbe2")
    ax.grid(axis="y", color="#36506e", alpha=0.28)
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles[:2],
        labels[:2],
        facecolor="#0f2238",
        edgecolor="#274261",
        labelcolor="#eaf3ff",
    )
    for text in legend.get_texts():
        text.set_color("#eaf3ff")
    for spine in ax.spines.values():
        spine.set_color("#274261")

    return fig


def generate_failure_gauge(failure_probability):
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    fig.patch.set_facecolor("#0b1a2c")
    ax.set_facecolor("#0f2238")

    remaining = max(1 - failure_probability, 0)
    colors = [get_risk_color_by_probability(failure_probability), "#20364f"]
    ax.pie(
        [failure_probability, remaining],
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.24, "edgecolor": "#0f2238"},
    )
    ax.text(0, 0.10, "Failure Risk", ha="center", va="center", color="#8ea6c6", fontsize=11)
    ax.text(
        0,
        -0.08,
        f"{failure_probability * 100:.1f}%",
        ha="center",
        va="center",
        color="#f7fbff",
        fontsize=23,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    return fig


def generate_class_distribution_chart(dataset):
    if "Architecture_Risk_Class" not in dataset.columns:
        return None

    counts = dataset["Architecture_Risk_Class"].value_counts()
    labels = counts.index.tolist()
    colors = [get_class_color(label) for label in labels]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fig.patch.set_facecolor("#0b1a2c")
    ax.set_facecolor("#0f2238")
    wedges, _ = ax.pie(
        counts.values,
        labels=None,
        startangle=100,
        colors=colors,
        wedgeprops={"width": 0.38, "edgecolor": "#0f2238"},
    )
    ax.legend(
        wedges,
        [f"{label}: {value}" for label, value in zip(labels, counts.values)],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        labelcolor="#eaf3ff",
    )
    ax.set_title("Leak Class Mix", color="#eaf3ff", fontsize=15, pad=14)
    ax.set_aspect("equal")
    return fig


inject_styles()
sns.set_theme(style="white")

try:
    model_bundle = load_ml_assets()
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_names = model_bundle["feature_names"]
    raw_input_features = model_bundle.get("raw_input_features", RAW_INPUT_FEATURES)
    regressors = model_bundle.get("regressors", {})
    integrated_dataset = load_integrated_dataset()
    analysis_summary = load_analysis_summary()
except Exception as exc:
    st.error(
        "Error loading AI models. Please ensure `train_models.py` has been executed. "
        f"({exc})"
    )
    st.stop()


st.sidebar.markdown("## Input Console")
st.sidebar.caption("Shape the field conditions and operational load for the current asset.")

scenario_mode = st.sidebar.radio(
    "Scenario source",
    ["Manual sliders", "Integrated dataset sample", "Upload CSV"],
    index=1,
)

selected_sample = None
if scenario_mode == "Integrated dataset sample":
    sample_options = {
        make_display_name(row, idx): idx
        for idx, row in integrated_dataset.head(250).iterrows()
    }
    selected_label = st.sidebar.selectbox(
        "Choose a field scenario",
        options=list(sample_options.keys()),
    )
    selected_sample = integrated_dataset.loc[sample_options[selected_label]]
    st.sidebar.caption(
        "This preset is loaded from `data/rspm_integrated.csv`, which contains the "
        "feature rows used by the model pipeline."
    )
elif scenario_mode == "Upload CSV":
    st.sidebar.caption(
        "Upload row-based scenario data. The summary file "
        "`RSPM_Integrated_Analysis.csv` is not valid for prediction."
    )
    uploaded_file = st.sidebar.file_uploader(
        "Upload scenario CSV",
        type=["csv"],
        help="Upload a CSV with the base Layer 1 input columns. "
        "Integrated Layer 2 features are derived automatically.",
    )
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        is_valid, missing_columns = validate_uploaded_dataset(uploaded_df, raw_input_features)
        if is_valid:
            uploaded_df = augment_with_architecture(uploaded_df)
            uploaded_options = {
                make_display_name(row, idx): idx
                for idx, row in uploaded_df.iterrows()
            }
            selected_uploaded_label = st.sidebar.selectbox(
                "Choose uploaded row",
                options=list(uploaded_options.keys()),
            )
            selected_sample = uploaded_df.loc[uploaded_options[selected_uploaded_label]]
            st.sidebar.success(f"Uploaded {len(uploaded_df)} scenario rows.")
        else:
            if looks_like_summary_csv(uploaded_df):
                st.sidebar.error(
                    "This looks like the summary report CSV, not the model input dataset."
                )
                st.sidebar.caption(
                    "Use `data/rspm_integrated.csv` for real sample rows, or upload a CSV "
                    "where each row is one scenario with the feature columns below."
                )
            else:
                st.sidebar.error(
                    "The uploaded CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )
            template_df = build_upload_template(raw_input_features)
            st.sidebar.download_button(
                "Download upload template",
                template_df.to_csv(index=False).encode("utf-8"),
                file_name="RSPM_Upload_Template.csv",
                mime="text/csv",
            )
            st.sidebar.caption("Expected columns: " + ", ".join(raw_input_features))

with st.sidebar:
    st.markdown("### Reservoir and Production")
    res_pressure = st.slider(
        "Reservoir Pressure (psi)",
        3000.0,
        8000.0,
        float(selected_sample["Reservoir_Pressure_psi"]) if selected_sample is not None else 4500.0,
    )
    res_temp = st.slider(
        "Reservoir Temperature (deg C)",
        60.0,
        150.0,
        float(selected_sample["Reservoir_Temperature_C"]) if selected_sample is not None else 85.0,
    )
    oil_rate = st.slider(
        "Oil Production Rate (bbl/day)",
        1000.0,
        50000.0,
        float(selected_sample["Oil_Production_Rate_bbl_day"]) if selected_sample is not None else 15000.0,
    )
    gas_rate = st.slider(
        "Gas Production Rate (MSCF/day)",
        500.0,
        20000.0,
        float(selected_sample["Gas_Production_Rate_MSCF_day"]) if selected_sample is not None else 8000.0,
    )
    water_cut = st.slider(
        "Water Cut (%)",
        0.0,
        90.0,
        float(selected_sample["Water_Cut_percent"]) if selected_sample is not None else 30.0,
    )

    st.markdown("### Pipeline Geometry")
    pipeline_diameter = st.slider(
        "Pipeline Diameter (m)",
        0.1,
        1.0,
        float(selected_sample["Pipeline_Diameter_m"]) if selected_sample is not None else 0.4,
    )
    wall_thickness = st.slider(
        "Wall Thickness (mm)",
        5.0,
        25.0,
        float(selected_sample["Wall_Thickness_mm"]) if selected_sample is not None else 10.0,
    )
    pipeline_length = st.slider(
        "Pipeline Length (km)",
        5.0,
        100.0,
        float(selected_sample["Pipeline_Length_km"]) if selected_sample is not None else 35.0,
    )
    flow_velocity = st.slider(
        "Flow Velocity (m/s)",
        0.5,
        5.0,
        float(selected_sample["Flow_Velocity_m_s"]) if selected_sample is not None else 2.5,
    )
    fluid_density = st.slider(
        "Fluid Density (kg/m^3)",
        700.0,
        1000.0,
        float(selected_sample["Fluid_Density_kg_m3"]) if selected_sample is not None else 850.0,
    )
    fluid_viscosity = st.slider(
        "Fluid Viscosity (cP)",
        1.0,
        50.0,
        float(selected_sample["Fluid_Viscosity_cP"]) if selected_sample is not None else 12.0,
    )

    st.markdown("### Integrity Factors")
    corrosion_rate = st.slider(
        "Corrosion Rate (mm/year)",
        0.01,
        2.0,
        float(selected_sample["Corrosion_Rate_mm_year"]) if selected_sample is not None else 0.25,
    )
    internal_pressure = st.slider(
        "Internal Pressure (psi)",
        500.0,
        3000.0,
        float(selected_sample["Internal_Pressure_psi"]) if selected_sample is not None else 1200.0,
    )
    temp_gradient = st.slider(
        "Temperature Gradient (deg C/km)",
        0.5,
        5.0,
        float(selected_sample["Temperature_Gradient_C_km"]) if selected_sample is not None else 1.5,
    )
    elevation_change = st.slider(
        "Elevation Change (m)",
        -200.0,
        200.0,
        float(selected_sample["Elevation_Change_m"]) if selected_sample is not None else -10.0,
    )
    pipeline_age = st.slider(
        "Pipeline Age (years)",
        1.0,
        40.0,
        float(selected_sample["Pipeline_Age_years"]) if selected_sample is not None else 15.0,
    )


values = {
    "res_pressure": res_pressure,
    "res_temp": res_temp,
    "oil_rate": oil_rate,
    "gas_rate": gas_rate,
    "water_cut": water_cut,
    "pipeline_diameter": pipeline_diameter,
    "wall_thickness": wall_thickness,
    "pipeline_length": pipeline_length,
    "flow_velocity": flow_velocity,
    "fluid_density": fluid_density,
    "fluid_viscosity": fluid_viscosity,
    "corrosion_rate": corrosion_rate,
    "internal_pressure": internal_pressure,
    "temp_gradient": temp_gradient,
    "elevation_change": elevation_change,
    "pipeline_age": pipeline_age,
}


failure_pressure = calculate_failure_pressure(
    pipeline_diameter, wall_thickness, corrosion_rate, pipeline_age
)
failure_probability = calculate_failure_probability(
    internal_pressure, failure_pressure, corrosion_rate, pipeline_age
)
architecture_summary = summarize_architecture(values)
lrs_score = architecture_summary["lrs_score"]
architecture_risk = architecture_summary["risk_class"]
decision_label = architecture_summary["decision_label"]
decision_action = architecture_summary["decision_action"]
pressure_ratio = architecture_summary["pressure_ratio"]
integrity_margin = architecture_summary["integrity_margin"]
baseline_health = determine_pipeline_health_status("Low", failure_probability)
health_class, health_accent = get_health_style(baseline_health)

st.markdown(
    f"""
    <div class="hero">
        <span class="eyebrow">Reservoir and Surface Pipeline Monitoring</span>
        <h1>Pipeline risk intelligence with a control-room feel.</h1>
        <p>
            Blend first-principles engineering with your trained ML classifier to
            understand structural margin, failure probability, and intervention urgency
            in one operational view.
        </p>
        <div class="hero-grid">
            <div class="mini-tile">
                <div class="mini-tile-label">Architecture LRS</div>
                <div class="mini-tile-value">{lrs_score:.1f}</div>
            </div>
            <div class="mini-tile">
                <div class="mini-tile-label">4-Class Decision</div>
                <div class="mini-tile-value">{decision_label}</div>
            </div>
            <div class="mini-tile">
                <div class="mini-tile-label">Required action</div>
                <div class="mini-tile-value">{decision_action}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_section_header(
    "Architecture Compliance",
    "The live scenario now follows the requested 4-layer flow: Layer 1 inputs, Layer 2 integrated features, Layer 3 AI outputs, and Layer 4 decision action.",
)
architecture_left, architecture_mid, architecture_right = st.columns(3)

with architecture_left:
    render_stat_card(
        "Layer 2 integrated features",
        f"{len(INTEGRATED_FEATURES)}",
        "10 reservoir-oriented + 12 pipeline-oriented engineered features.",
        "#4cc9f0",
    )
with architecture_mid:
    render_stat_card(
        "Layer 3 LRS score",
        f"{lrs_score:.1f}",
        "Computed using FRD, CR, PDA, WC, GOR, and PU weightings from the architecture diagram.",
        get_class_color(architecture_risk),
    )
with architecture_right:
    render_stat_card(
        "Layer 4 action",
        decision_action,
        f"{decision_label} response derived from the combined classifier and LRS thresholds.",
        get_class_color(decision_label),
    )

component_frame = pd.DataFrame(
    [{"Component": key, "Score": round(value, 2)} for key, value in architecture_summary["components"].items()]
)
st.markdown('<div class="table-frame">', unsafe_allow_html=True)
st.dataframe(component_frame, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

if analysis_summary is not None:
    render_section_header(
        "Integrated Analysis Brief",
        "Reference the summary exported in `RSPM_Integrated_Analysis.csv` alongside the live dashboard scenario.",
    )
    analysis_col, dataset_col = st.columns([1.2, 1])
    with analysis_col:
        st.markdown('<div class="table-frame">', unsafe_allow_html=True)
        st.dataframe(analysis_summary, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
with dataset_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Dataset Coverage")
    st.caption("The summary CSV is descriptive, while inference rows come from the integrated dataset.")
    class_mix = (
        integrated_dataset["Architecture_Risk_Class"].value_counts(normalize=True).reindex(DECISION_ACTIONS.keys(), fill_value=0) * 100
    )
    for class_name, pct in class_mix.items():
        st.metric(f"{class_name} share", f"{pct:.1f}%")
    st.metric("Available scenario rows", f"{len(integrated_dataset):,}")
    class_mix_chart = generate_class_distribution_chart(integrated_dataset)
    if class_mix_chart is not None:
        st.pyplot(class_mix_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


render_section_header(
    "Engineering Snapshot",
    "A quick read of containment strength, degradation exposure, and current operating headroom.",
)
stat_cols = st.columns(4)
with stat_cols[0]:
    render_stat_card(
        "Failure pressure",
        f"{failure_pressure:,.0f} psi",
        "Calculated with Barlow-based wall strength and corrosion-adjusted thickness.",
        "#4cc9f0",
    )
with stat_cols[1]:
    render_stat_card(
        "Failure probability",
        f"{failure_probability * 100:.1f}%",
        "Pressure ratio, corrosion, and asset age are blended into a bounded risk score.",
        get_risk_color_by_probability(failure_probability),
    )
with stat_cols[2]:
    render_stat_card(
        "Integrity margin",
        f"{integrity_margin:,.0f} psi",
        "Headroom between current internal pressure and estimated structural failure pressure.",
        "#f7b267",
    )
with stat_cols[3]:
    render_stat_card(
        "Baseline health",
        baseline_health.split(" - ")[0],
        baseline_health,
        health_accent,
    )


overview_left, overview_right = st.columns([1.4, 1])
with overview_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Pressure Envelope")
    st.caption("Internal pressure tracked against the estimated failure limit for the configured asset.")
    pressure_frame = pd.DataFrame(
        {
            "Metric": ["Internal pressure", "Failure pressure"],
            "Pressure (psi)": [internal_pressure, failure_pressure],
        }
    )
    fig_pressure, ax_pressure = plt.subplots(figsize=(8.6, 3.1))
    fig_pressure.patch.set_facecolor("#0b1a2c")
    ax_pressure.set_facecolor("#0f2238")
    ax_pressure.barh(
        pressure_frame["Metric"],
        pressure_frame["Pressure (psi)"],
        color=["#f7b267", "#4cc9f0"],
        height=0.55,
    )
    ax_pressure.set_xlabel("Pressure (psi)", color="#9fb6d3")
    ax_pressure.tick_params(colors="#b9cbe2")
    ax_pressure.grid(axis="x", color="#36506e", alpha=0.28)
    for spine in ax_pressure.spines.values():
        spine.set_color("#274261")
    st.pyplot(fig_pressure, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with overview_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Risk Gauge")
    st.caption("Current scenario expressed against the leak-severity palette.")
    gauge_chart = generate_failure_gauge(failure_probability)
    st.pyplot(gauge_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


render_section_header(
    "AI Risk Diagnosis",
    "Run the architecture-aligned AI engine: Layer 2 feature integration, Layer 3 classification and regression, then Layer 4 actioning.",
)

input_data = build_input_frame(values)
integrated_input_data = build_integrated_feature_frame(input_data)
integrated_input_data = integrated_input_data[feature_names]

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if st.button("Run risk diagnosis"):
    scaled_input = scaler.transform(integrated_input_data)
    prediction_class = model.predict(scaled_input)[0]
    predicted_lrs = (
        float(regressors["lrs"].predict(scaled_input)[0])
        if "lrs" in regressors
        else lrs_score
    )
    predicted_corrosion = (
        float(regressors["corrosion_rate"].predict(scaled_input)[0])
        if "corrosion_rate" in regressors
        else values["corrosion_rate"]
    )
    predicted_failure_pressure = (
        float(regressors["failure_pressure"].predict(scaled_input)[0])
        if "failure_pressure" in regressors
        else failure_pressure
    )
    decision_output = resolve_decision_output(
        prediction_class,
        predicted_lrs,
        predicted_corrosion,
        pressure_ratio,
    )
    health_status = determine_pipeline_health_status(prediction_class, failure_probability)
    export_payload = input_data.iloc[0].to_dict()
    export_payload.update(integrated_input_data.iloc[0].to_dict())
    export_payload["Predicted_Leak_Risk_Class"] = prediction_class
    export_payload["Predicted_LRS_Score"] = round(predicted_lrs, 2)
    export_payload["Predicted_Corrosion_Rate_mm_year"] = round(predicted_corrosion, 4)
    export_payload["Predicted_Failure_Pressure_psi"] = round(predicted_failure_pressure, 2)
    export_payload["Decision_Output"] = decision_output["decision_label"]
    export_payload["Recommended_Action"] = decision_output["decision_action"]
    export_payload["Failure_Probability_Percent"] = f"{failure_probability * 100:.2f}%"
    export_payload["Infrastructure_Health_Status"] = health_status

    st.session_state.latest_result = {
        "prediction_class": prediction_class,
        "predicted_lrs": predicted_lrs,
        "predicted_corrosion": predicted_corrosion,
        "predicted_failure_pressure": predicted_failure_pressure,
        "decision_output": decision_output,
        "health_status": health_status,
        "export_payload": export_payload,
    }


diagnosis_left, diagnosis_right = st.columns([1.25, 1])
with diagnosis_left:
    if st.session_state.latest_result:
        result = st.session_state.latest_result
        banner_class, _ = get_health_style(result["health_status"])
        st.markdown(
            f"""
            <div class="risk-banner {banner_class}">
                <h3>{result["decision_output"]["decision_label"]} decision output</h3>
                <p>
                    Layer 3 classification: {result["prediction_class"]} |
                    Layer 3 LRS: {result["predicted_lrs"]:.1f} |
                    Action: {result["decision_output"]["decision_action"]}
                </p>
                <div class="status-pill">Current failure probability: {failure_probability * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0;">Diagnosis awaiting model run</h3>
                <p style="color:#8ea6c6;margin-bottom:0;">
                    Tune the operating conditions in the sidebar, then run the diagnosis to
                    produce a leak risk class and downloadable reports.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with diagnosis_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Export Console")
    st.caption("Generate stakeholder-ready report files from the most recent diagnosis.")
    if st.session_state.latest_result:
        excel_path = generate_excel_report(st.session_state.latest_result["export_payload"])
        pdf_path = generate_pdf_report(st.session_state.latest_result["export_payload"])

        with open(excel_path, "rb") as file_excel:
            st.download_button(
                "Download Excel report",
                file_excel,
                file_name="RSPM_Risk_Assessment.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with open(pdf_path, "rb") as file_pdf:
            st.download_button(
                "Download PDF report",
                file_pdf,
                file_name="RSPM_Risk_Assessment.pdf",
                mime="application/pdf",
            )
    else:
        st.info("Run the diagnosis first to enable exports.")
    st.markdown("</div>", unsafe_allow_html=True)


render_section_header(
    "Risk Projection and Drivers",
    "See how the asset degrades over time and which live operating variables are currently carrying the most weight.",
)
projection_col, driver_col = st.columns([1.45, 1])

with projection_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    projection_chart = generate_projection_chart(values)
    st.pyplot(projection_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with driver_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    risk_profile = make_risk_profile(values, failure_probability, pressure_ratio)
    driver_chart = generate_driver_chart(risk_profile)
    st.pyplot(driver_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


render_section_header(
    "Parameter Categories and Analysis",
    "Group the current scenario into reservoir, pipeline, and fluid-chemistry views, then plot the operational profile.",
)
parameter_categories = build_parameter_categories(values, failure_pressure)
category_col, comparison_col = st.columns([1.2, 1])

with category_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Categorised Parameters")
    st.caption(
        "CO2 content is not available in the current dataset. H2O content is represented by water cut as a practical proxy."
    )
    display_categories = parameter_categories.copy()
    display_categories["Value"] = display_categories["Value"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value:,.2f}"
    )
    st.dataframe(display_categories, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with comparison_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    category_chart = generate_category_chart(parameter_categories)
    st.pyplot(category_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
comparison_chart = generate_pipeline_reservoir_comparison(values, failure_pressure)
st.pyplot(comparison_chart, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


render_section_header(
    "Scenario Snapshot",
    "A compact view of the 22 Layer 2 integrated features passed into the architecture-compliant AI engine.",
)
display_frame = integrated_input_data.T.reset_index()
display_frame.columns = ["Feature", "Value"]
st.markdown('<div class="table-frame">', unsafe_allow_html=True)
st.dataframe(display_frame, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)
