"""
South Sudan Field Guardian (PFG AI) - Enhanced Dashboard
Production-ready with custom HTML/CSS/JS for modern UI
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PFG AI Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🛡️"
)

# Custom CSS for modern dark theme
def inject_custom_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        }
        
        /* Custom header */
        .custom-header {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        .custom-header h1 {
            color: #ffffff;
            font-size: 2.5em;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .custom-header p {
            color: rgba(255, 255, 255, 0.7);
            font-size: 1.1em;
            margin-top: 10px;
        }
        
        /* Metric cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: 700;
            color: #ffffff;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: rgba(255, 255, 255, 0.6);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-change {
            font-size: 0.9em;
            padding: 5px 10px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 10px;
        }
        
        /* Alert styles */
        .alert-critical {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            font-weight: 500;
            box-shadow: 0 8px 32px 0 rgba(245, 87, 108, 0.3);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        /* Table styles */
        .dataframe {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }
        
        .dataframe th {
            background: rgba(102, 126, 234, 0.3) !important;
            color: white !important;
            padding: 15px !important;
            font-weight: 600 !important;
        }
        
        .dataframe td {
            color: rgba(255, 255, 255, 0.9) !important;
            padding: 12px !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Button styles */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
        }
        
        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-weight: 600;
            width: 100%;
        }
        
        /* Card container */
        .card-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 15px 0;
        }
        
        /* Section titles */
        .section-title {
            color: #ffffff;
            font-size: 1.8em;
            font-weight: 600;
            margin: 30px 0 20px 0;
            padding-left: 15px;
            border-left: 4px solid #667eea;
        }
        
        /* Status badge */
        .status-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            display: inline-block;
        }
        
        .status-critical {
            background: rgba(244, 67, 54, 0.3);
            color: #ff5252;
            border: 1px solid #ff5252;
        }
        
        .status-medium {
            background: rgba(255, 152, 0, 0.3);
            color: #ffa726;
            border: 1px solid #ffa726;
        }
        
        .status-safe {
            background: rgba(76, 175, 80, 0.3);
            color: #66bb6a;
            border: 1px solid #66bb6a;
        }
        
        /* Recommendation card */
        .recommendation-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-left: 4px solid #667eea;
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 8px;
            color: rgba(255, 255, 255, 0.9);
        }
        
        /* Progress bar */
        .progress-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.5);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(102, 126, 234, 0.7);
        }
    </style>
    """, unsafe_allow_html=True)

# Configuration
class Config:
    DEFAULT_DATA_PATH = "dataset_rspm_paloch.csv"
    FEATURE_COLS = [
        'pressure_bar', 'pressure_drop_30d', 'pressure_decline', 'q_bbl_day', 'dprod_30d',
        'water_cut', 'gas_oil_ratio', 'co2_pct', 'h2s_pct',
        'pipe_diameter_m', 'wall_thickness_mm', 'pipe_age_yrs',
        'segment_length_km', 'corrosion_mm_yr', 'rul_days'
    ]
    RISK_THRESHOLDS = {'low': 0.33, 'high': 0.66}
    MODEL_PARAMS = {
        'n_estimators': 200,
        'max_depth': 12,
        'random_state': 42,
        'n_jobs': -1
    }

# Data loading and model training functions
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(path: str) -> pd.DataFrame:
    try:
        if not Path(path).exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if 'pressure_decline' not in df.columns:
            if 'pressure_drop_30d' in df.columns:
                df['pressure_decline'] = df['pressure_drop_30d'] / 30.0
            else:
                df['pressure_decline'] = 0.0
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame, feature_cols: list, target_col: str = 'leak_flag'):
    try:
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].fillna(0.0)
        if target_col not in df.columns:
            return None, None, None
        y = df[target_col].fillna(0).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        clf = RandomForestClassifier(**Config.MODEL_PARAMS)
        clf.fit(X_train, y_train)
        metrics = {}
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
            if len(np.unique(y_test)) > 1:
                metrics['auc'] = roc_auc_score(y_test, y_proba)
        except:
            metrics['auc'] = None
        feature_importance = dict(zip(available_cols, clf.feature_importances_))
        return clf, metrics, feature_importance
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None, None

def categorize_risk(probability: float) -> str:
    if probability < Config.RISK_THRESHOLDS['low']:
        return 'Safe'
    elif probability < Config.RISK_THRESHOLDS['high']:
        return 'Medium'
    else:
        return 'Critical'

def get_status_badge(risk_level: str) -> str:
    badge_class = {
        'Critical': 'status-critical',
        'Medium': 'status-medium',
        'Safe': 'status-safe'
    }
    return f'<span class="status-badge {badge_class[risk_level]}">{risk_level}</span>'

def recommend_action(row: pd.Series) -> list:
    p = row.get('predicted_leak_prob', 0)
    recs = []
    if p >= Config.RISK_THRESHOLDS['high']:
        recs.append("🚨 URGENT: Schedule ultrasonic inspection and implement daily monitoring")
    elif p >= Config.RISK_THRESHOLDS['low']:
        recs.append("⚠️ MEDIUM: Schedule visual & non-intrusive inspection (weekly)")
    else:
        recs.append("✅ LOW: Maintain routine inspection schedule")
    if row.get('corrosion_mm_yr', 0) > 0.08:
        recs.append("🔧 High corrosion rate: Apply corrosion inhibition or schedule pigging")
    if row.get('water_cut', 0) > 0.7:
        recs.append("💧 High water cut: Review water management systems")
    if row.get('pressure_decline', 0) < -0.5:
        recs.append("📉 Rapid pressure decline: Conduct reservoir diagnostic")
    return recs

# Main application
def main():
    inject_custom_css()
    
    # Custom header
    st.markdown("""
    <div class="custom-header">
        <h1> South Sudan Oil Field Guardian</h1>
        <p>AI-Powered Pipeline Integrity & Leak Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for data loading
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        upload_option = st.radio("Data Source", ["File Path", "Upload CSV"])
        
        if upload_option == "Upload CSV":
            uploaded_file = st.file_uploader("Upload Dataset", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                st.info("Upload a CSV file to begin")
                return
        else:
            data_path = st.text_input("CSV File Path", value=Config.DEFAULT_DATA_PATH)
            df = load_data(data_path)
            if df.empty:
                st.error("❌ File not found")
                st.stop()
        
        st.markdown("---")
        st.metric("Dataset Size", f"{len(df):,} rows")
        
        if st.button("🔄 Retrain Model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Train model
    with st.spinner("Training model..."):
        model, metrics, feature_importance = train_model(df, Config.FEATURE_COLS)
    
    if model is None:
        st.error("❌ Model training failed")
        st.stop()
    
    # Generate predictions
    available_cols = [col for col in Config.FEATURE_COLS if col in df.columns]
    df['predicted_leak_prob'] = model.predict_proba(df[available_cols].fillna(0.0))[:, 1]
    df['risk_level'] = df['predicted_leak_prob'].apply(categorize_risk)
    
    # Calculate metrics
    critical_count = int((df['risk_level'] == 'Critical').sum())
    medium_count = int((df['risk_level'] == 'Medium').sum())
    safe_count = int((df['risk_level'] == 'Safe').sum())
    max_prob = float(df['predicted_leak_prob'].max())
    avg_prob = float(df['predicted_leak_prob'].mean())
    
    # Key Metrics Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔴 Critical Segments</div>
            <div class="metric-value">{critical_count:,}</div>
            <div class="metric-change" style="background: rgba(244, 67, 54, 0.2); color: #ff5252;">
                {(critical_count/len(df)*100):.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🟠 Medium Risk</div>
            <div class="metric-value">{medium_count:,}</div>
            <div class="metric-change" style="background: rgba(255, 152, 0, 0.2); color: #ffa726;">
                {(medium_count/len(df)*100):.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🟢 Safe Segments</div>
            <div class="metric-value">{safe_count:,}</div>
            <div class="metric-change" style="background: rgba(76, 175, 80, 0.2); color: #66bb6a;">
                {(safe_count/len(df)*100):.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ Max Risk Score</div>
            <div class="metric-value">{max_prob:.3f}</div>
            <div class="metric-change" style="background: rgba(102, 126, 234, 0.2); color: #667eea;">
                Avg: {avg_prob:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Critical Alert
    if critical_count > 0:
        st.markdown(f"""
        <div class="alert-critical">
            <strong>⚠️ CRITICAL ALERT</strong><br>
            {critical_count} segments require immediate attention! Review the critical segments table below for detailed analysis and recommendations.
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive Risk Map with Plotly
    st.markdown('<h2 class="section-title">📊 Interactive Risk Map</h2>', unsafe_allow_html=True)
    
    if 'segment_length_km' in df.columns:
        color_map = {'Safe': '#66bb6a', 'Medium': '#ffa726', 'Critical': '#ff5252'}
        df['color'] = df['risk_level'].map(color_map)
        
        fig = go.Figure()
        
        for risk in ['Safe', 'Medium', 'Critical']:
            df_risk = df[df['risk_level'] == risk]
            fig.add_trace(go.Scatter(
                x=df_risk['segment_length_km'],
                y=df_risk['predicted_leak_prob'],
                mode='markers',
                name=risk,
                marker=dict(
                    size=10,
                    color=color_map[risk],
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>Location:</b> %{x:.2f} km<br>' +
                             '<b>Risk Score:</b> %{y:.3f}<br>' +
                             f'<b>Level:</b> {risk}<extra></extra>'
            ))
        
        fig.add_hline(y=Config.RISK_THRESHOLDS['low'], line_dash="dash", 
                     line_color="rgba(255, 167, 38, 0.5)", annotation_text="Medium Threshold")
        fig.add_hline(y=Config.RISK_THRESHOLDS['high'], line_dash="dash", 
                     line_color="rgba(255, 82, 82, 0.5)", annotation_text="Critical Threshold")
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            xaxis_title="Pipeline Location (km)",
            yaxis_title="Predicted Leak Probability",
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Two column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<h2 class="section-title">🚨 Critical Segments</h2>', unsafe_allow_html=True)
        
        top_critical = df[df['risk_level'] == 'Critical'].sort_values(
            'predicted_leak_prob', ascending=False
        ).head(10)
        
        if len(top_critical) > 0:
            display_cols = ['segment_length_km', 'predicted_leak_prob', 'risk_level', 
                          'corrosion_mm_yr', 'rul_days']
            display_cols = [col for col in display_cols if col in top_critical.columns]
            
            st.dataframe(
                top_critical[display_cols].style.format({
                    'predicted_leak_prob': '{:.3f}',
                    'segment_length_km': '{:.2f}',
                    'corrosion_mm_yr': '{:.4f}',
                    'rul_days': '{:.0f}'
                }),
                use_container_width=True,
                height=400
            )
        else:
            st.success("✅ No critical segments detected!")
    
    with col_right:
        st.markdown('<h2 class="section-title">🎯 Feature Importance</h2>', unsafe_allow_html=True)
        
        if feature_importance:
            fi_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
            features, importances = zip(*fi_sorted)
            
            fig_fi = go.Figure(go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker=dict(
                    color=list(importances),
                    colorscale='Viridis',
                    line=dict(color='white', width=1)
                )
            ))
            
            fig_fi.update_layout(
                template='plotly_dark',
                height=400,
                xaxis_title="Importance",
                yaxis_title="",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False
            )
            
            st.plotly_chart(fig_fi, use_container_width=True)
    
    # Segment Inspector
    st.markdown('<h2 class="section-title">🔬 Segment Inspector</h2>', unsafe_allow_html=True)
    
    if 'segment_length_km' in df.columns:
        col_inspect1, col_inspect2 = st.columns([1, 2])
        
        with col_inspect1:
            km_choice = st.number_input(
                "Enter Pipeline Location (km)",
                min_value=float(df['segment_length_km'].min()),
                max_value=float(df['segment_length_km'].max()),
                value=float(df['segment_length_km'].iloc[0]),
                step=0.1
            )
        
        nearest_idx = (df['segment_length_km'] - km_choice).abs().idxmin()
        segment = df.loc[nearest_idx]
        
        with col_inspect2:
            st.markdown(f"""
            <div class="card-container">
                <h3>Segment Details</h3>
                <p><strong>Location:</strong> {segment['segment_length_km']:.2f} km</p>
                <p><strong>Risk Score:</strong> {segment['predicted_leak_prob']:.3f}</p>
                <p><strong>Status:</strong> {get_status_badge(segment['risk_level'])}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 AI-Generated Recommendations")
        recommendations = recommend_action(segment)
        for rec in recommendations:
            st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
    
    # Export Section
    st.markdown('<h2 class="section-title">📥 Export Data</h2>', unsafe_allow_html=True)
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        critical_csv = df[df['risk_level'] == 'Critical'].to_csv(index=False)
        st.download_button(
            label=f"📄 Download Critical Segments ({critical_count} rows)",
            data=critical_csv,
            file_name=f"critical_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        full_csv = df.to_csv(index=False)
        st.download_button(
            label=f"📊 Download Full Dataset ({len(df):,} rows)",
            data=full_csv,
            file_name=f"full_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: rgba(255, 255, 255, 0.5); padding: 40px 0 20px 0; margin-top: 50px; border-top: 1px solid rgba(255, 255, 255, 0.1);'>
        <p><strong>South Sudan Field Guardian (PFG AI)</strong> v2.0 | Enhanced Dashboard</p>
        <p style='font-size: 0.9em;'>Production-Ready AI System for Pipeline Integrity Management</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)