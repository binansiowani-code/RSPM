"""
Paloch Field Guardian (PFG AI) - Production Ready
Optimized for local deployment and Streamlit hosting
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import io
import logging
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Paloch Field Guardian (PFG AI)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .critical-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 10px 0;
    }
    .stDownloadButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Configuration
# -----------------------
class Config:
    """Application configuration"""
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
        'n_jobs': -1,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }

# -----------------------
# Utilities & Data Processing
# -----------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess dataset with error handling"""
    try:
        if not Path(path).exists():
            st.error(f"❌ File not found: {path}")
            st.info("Please ensure the CSV file is in the same directory as the app.")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        logger.info(f"Loaded dataset: {len(df)} rows × {df.shape[1]} columns")
        
        # Compute pressure_decline if missing
        if 'pressure_decline' not in df.columns:
            if 'pressure_drop_30d' in df.columns:
                df['pressure_decline'] = df['pressure_drop_30d'] / 30.0
                logger.info("Computed 'pressure_decline' from 'pressure_drop_30d'")
            else:
                df['pressure_decline'] = 0.0
                logger.warning("'pressure_decline' set to 0 - missing source data")
        
        # Validate required columns
        missing_cols = [col for col in Config.FEATURE_COLS if col not in df.columns]
        if missing_cols:
            st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            logger.warning(f"Missing columns: {missing_cols}")
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        logger.error(f"Data loading error: {e}", exc_info=True)
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame, feature_cols: list, target_col: str = 'leak_flag'):
    """Train RandomForest model with comprehensive metrics"""
    try:
        # Prepare features and target
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].fillna(0.0)
        
        if target_col not in df.columns:
            st.error(f"❌ Target column '{target_col}' not found in dataset")
            return None, None, None, None
        
        y = df[target_col].fillna(0).astype(int)
        
        # Check for sufficient data
        if len(X) < 10:
            st.error("❌ Insufficient data for training (minimum 10 samples required)")
            return None, None, None, None
        
        # Train-test split with stratification
        test_size = 0.2
        stratify = y if len(np.unique(y)) > 1 and y.sum() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        
        # Train model
        logger.info("Training RandomForest model...")
        clf = RandomForestClassifier(**Config.MODEL_PARAMS)
        clf.fit(X_train, y_train)
        
        # Evaluate model
        metrics = {}
        try:
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            if len(np.unique(y_test)) > 1:
                metrics['auc'] = roc_auc_score(y_test, y_proba)
            else:
                metrics['auc'] = None
            
            metrics['accuracy'] = (y_pred == y_test).mean()
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
            logger.info(f"Model trained successfully - AUC: {metrics.get('auc', 'N/A')}")
        
        except Exception as e:
            logger.warning(f"Error computing metrics: {e}")
            metrics['auc'] = None
            metrics['accuracy'] = None
        
        # Feature importance
        feature_importance = dict(zip(available_cols, clf.feature_importances_))
        
        return clf, metrics, feature_importance, available_cols
    
    except Exception as e:
        st.error(f"❌ Model training error: {str(e)}")
        logger.error(f"Training error: {e}", exc_info=True)
        return None, None, None, None

def categorize_risk(probability: float) -> str:
    """Categorize risk level based on probability"""
    if probability < Config.RISK_THRESHOLDS['low']:
        return 'Safe'
    elif probability < Config.RISK_THRESHOLDS['high']:
        return 'Medium'
    else:
        return 'Critical'

def get_color_map() -> dict:
    """Get color mapping for risk levels"""
    return {'Safe': 'green', 'Medium': 'orange', 'Critical': 'red'}

def recommend_action(row: pd.Series) -> list:
    """Generate actionable recommendations based on segment data"""
    p = row.get('predicted_leak_prob', 0)
    recs = []
    
    # Primary recommendation based on leak probability
    if p >= Config.RISK_THRESHOLDS['high']:
        recs.append("🚨 URGENT: Schedule ultrasonic inspection and implement daily monitoring")
    elif p >= Config.RISK_THRESHOLDS['low']:
        recs.append("⚠️ MEDIUM: Schedule visual & non-intrusive inspection (weekly)")
    else:
        recs.append("✅ LOW: Maintain routine inspection schedule")
    
    # Condition-specific recommendations
    if row.get('corrosion_mm_yr', 0) > 0.08:
        recs.append("🔧 High corrosion rate detected: Apply corrosion inhibition or schedule pigging")
    
    if row.get('water_cut', 0) > 0.7:
        recs.append("💧 High water cut: Review water management and inlet separator performance")
    
    if row.get('pressure_decline', 0) < -0.5:
        recs.append("📉 Rapid pressure decline: Conduct reservoir diagnostic and well test")
    
    if row.get('rul_days', float('inf')) < 90:
        recs.append("⏰ Low RUL: Priority replacement or repair within 3 months")
    
    if row.get('h2s_pct', 0) > 5:
        recs.append("☠️ High H2S content: Ensure proper safety protocols and materials")
    
    return recs

# -----------------------
# Main Application
# -----------------------
def main():
    """Main application logic"""
    
    # Header
    st.title("🛡️ Paloch Field Guardian — PFG AI")
    st.markdown("""
    An integrated **Reservoir ↔ Pipeline AI** system for predicting leaks, corrosion hotspots,
    and generating actionable recommendations. Built for production-critical infrastructure monitoring.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload or path input
    upload_option = st.sidebar.radio("Data Source", ["File Path", "Upload CSV"])
    
    if upload_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Uploaded: {len(df):,} rows")
        else:
            st.info("👆 Please upload a CSV file to begin")
            return
    else:
        data_path = st.sidebar.text_input("CSV File Path", value=Config.DEFAULT_DATA_PATH)
        df = load_data(data_path)
        
        if df.empty:
            st.stop()
    
    # Dataset info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.metric("Total Rows", f"{len(df):,}")
    st.sidebar.metric("Total Columns", df.shape[1])
    
    # Model training section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Training")
    
    if st.sidebar.button("🔄 Retrain Model", type="primary"):
        st.cache_resource.clear()
        st.rerun()
    
    # Train or load model
    with st.spinner("🔧 Training model..."):
        model, metrics, feature_importance, used_features = train_model(
            df, Config.FEATURE_COLS
        )
    
    if model is None:
        st.error("❌ Model training failed. Please check your data.")
        st.stop()
    
    # Generate predictions
    with st.spinner("🔮 Generating predictions..."):
        try:
            X_pred = df[used_features].fillna(0.0)
            df['predicted_leak_prob'] = model.predict_proba(X_pred)[:, 1]
            df['risk_level'] = df['predicted_leak_prob'].apply(categorize_risk)
            df['color'] = df['risk_level'].map(get_color_map())
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.stop()
    
    # Display model metrics in sidebar
    if metrics:
        st.sidebar.markdown("### 📈 Model Performance")
        if metrics.get('auc'):
            st.sidebar.metric("ROC AUC", f"{metrics['auc']:.3f}")
        if metrics.get('accuracy'):
            st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    
    # Main dashboard layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "🗺️ Risk Map", "🔍 Segment Details", "📥 Export"
    ])
    
    # TAB 1: Overview
    with tab1:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        critical_count = int((df['risk_level'] == 'Critical').sum())
        medium_count = int((df['risk_level'] == 'Medium').sum())
        safe_count = int((df['risk_level'] == 'Safe').sum())
        
        with col1:
            st.metric("🔴 Critical Segments", f"{critical_count:,}", 
                     delta=f"{(critical_count/len(df)*100):.1f}%")
        with col2:
            st.metric("🟠 Medium Risk", f"{medium_count:,}",
                     delta=f"{(medium_count/len(df)*100):.1f}%")
        with col3:
            st.metric("🟢 Safe Segments", f"{safe_count:,}",
                     delta=f"{(safe_count/len(df)*100):.1f}%")
        with col4:
            st.metric("📊 Total Segments", f"{len(df):,}")
        
        # Additional metrics
        col5, col6 = st.columns(2)
        with col5:
            st.metric("⚠️ Max Leak Probability", f"{df['predicted_leak_prob'].max():.3f}")
        with col6:
            st.metric("📈 Avg Leak Probability", f"{df['predicted_leak_prob'].mean():.3f}")
        
        # Critical alert
        if critical_count > 0:
            st.markdown('<div class="critical-alert">', unsafe_allow_html=True)
            st.warning(f"🚨 **ALERT**: {critical_count} critical segments require immediate attention!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature importance
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("🎯 Top Feature Importance")
            if feature_importance:
                fi_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                fi_df = pd.DataFrame(fi_sorted, columns=['Feature', 'Importance'])
                fi_df['Importance'] = fi_df['Importance'].apply(lambda x: f"{x:.4f}")
                st.dataframe(fi_df.head(10), use_container_width=True, hide_index=True)
        
        with col_right:
            st.subheader("📊 Risk Distribution")
            risk_counts = df['risk_level'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = [get_color_map()[level] for level in risk_counts.index]
            ax.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7)
            ax.set_ylabel("Count")
            ax.set_title("Segments by Risk Level")
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    # TAB 2: Risk Map
    with tab2:
        st.subheader("🗺️ Leak Probability Risk Map")
        
        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            if 'segment_length_km' in df.columns:
                km_min, km_max = float(df['segment_length_km'].min()), float(df['segment_length_km'].max())
                km_range = st.slider("Pipeline Location (km)", 
                                    int(km_min), int(km_max), 
                                    (int(km_min), int(km_max)))
            else:
                km_range = (0, 100)
        
        with col_f2:
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                options=['Safe', 'Medium', 'Critical'],
                default=['Safe', 'Medium', 'Critical']
            )
        
        # Apply filters
        if 'segment_length_km' in df.columns:
            filtered_df = df[
                (df['segment_length_km'] >= km_range[0]) & 
                (df['segment_length_km'] <= km_range[1]) & 
                (df['risk_level'].isin(risk_filter))
            ]
        else:
            filtered_df = df[df['risk_level'].isin(risk_filter)]
        
        st.info(f"📌 Showing {len(filtered_df):,} of {len(df):,} segments")
        
        # Plot risk map
        if 'segment_length_km' in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(16, 6))
            scatter = ax.scatter(
                filtered_df['segment_length_km'],
                filtered_df['predicted_leak_prob'],
                c=filtered_df['color'],
                s=30,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            ax.set_xlabel("Pipeline Location (km)", fontsize=12)
            ax.set_ylabel("Predicted Leak Probability", fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3, linestyle='--')
            ax.axhline(y=Config.RISK_THRESHOLDS['low'], color='orange', linestyle='--', alpha=0.5, label='Medium Threshold')
            ax.axhline(y=Config.RISK_THRESHOLDS['high'], color='red', linestyle='--', alpha=0.5, label='Critical Threshold')
            
            # Legend
            patches = [mpatches.Patch(color=color, label=label) 
                      for label, color in get_color_map().items()]
            ax.legend(handles=patches, loc='upper right', fontsize=10)
            
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("⚠️ 'segment_length_km' column not found - cannot display map")
    
    # TAB 3: Segment Details
    with tab3:
        st.subheader("🔍 Segment Analysis & Recommendations")
        
        # Top critical segments
        st.markdown("### 🚨 Top Critical Segments")
        top_n = st.slider("Number of segments to display", 5, 50, 15)
        
        top_critical = df.sort_values('predicted_leak_prob', ascending=False).head(top_n)
        
        display_cols = ['scenario_id', 'day', 'segment_length_km', 'predicted_leak_prob', 
                       'risk_level', 'corrosion_mm_yr', 'rul_days']
        display_cols = [col for col in display_cols if col in top_critical.columns]
        
        st.dataframe(
            top_critical[display_cols].style.format({
                'predicted_leak_prob': '{:.3f}',
                'segment_length_km': '{:.2f}',
                'corrosion_mm_yr': '{:.4f}',
                'rul_days': '{:.0f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Individual segment inspection
        st.markdown("### 🔬 Individual Segment Inspection")
        
        if 'segment_length_km' in df.columns:
            km_choice = st.number_input(
                "Enter pipeline location (km)",
                min_value=float(df['segment_length_km'].min()),
                max_value=float(df['segment_length_km'].max()),
                value=float(df['segment_length_km'].iloc[0]),
                step=0.1
            )
            
            # Find nearest segment
            nearest_idx = (df['segment_length_km'] - km_choice).abs().idxmin()
            segment = df.loc[nearest_idx]
            
            # Display segment info
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Location (km)", f"{segment['segment_length_km']:.2f}")
            with col_b:
                st.metric("Leak Probability", f"{segment['predicted_leak_prob']:.3f}")
            with col_c:
                risk_color = get_color_map()[segment['risk_level']]
                st.markdown(f"**Risk Level:** <span style='color:{risk_color}; font-size:20px;'>●</span> {segment['risk_level']}", 
                           unsafe_allow_html=True)
            
            # Segment details
            st.markdown("#### 📋 Segment Parameters")
            detail_cols = ['pressure_bar', 'pressure_drop_30d', 'q_bbl_day', 
                          'water_cut', 'corrosion_mm_yr', 'rul_days', 'pipe_age_yrs']
            detail_cols = [col for col in detail_cols if col in segment.index]
            
            detail_data = segment[detail_cols].to_frame().T
            st.dataframe(detail_data, use_container_width=True)
            
            # Recommendations
            st.markdown("#### 💡 AI-Generated Recommendations")
            recommendations = recommend_action(segment)
            for rec in recommendations:
                st.info(rec)
    
    # TAB 4: Export
    with tab4:
        st.subheader("📥 Export & Download Options")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.markdown("#### 🔴 Critical Segments Report")
            critical_df = df[df['risk_level'] == 'Critical'].sort_values(
                'predicted_leak_prob', ascending=False
            )
            
            export_cols = [col for col in [
                'scenario_id', 'day', 'segment_length_km', 'predicted_leak_prob',
                'risk_level', 'corrosion_mm_yr', 'rul_days', 'pressure_bar',
                'water_cut', 'h2s_pct'
            ] if col in critical_df.columns]
            
            critical_csv = critical_df[export_cols].to_csv(index=False)
            
            st.download_button(
                label=f"📄 Download Critical Segments ({len(critical_df)} rows)",
                data=critical_csv,
                file_name=f"critical_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.metric("Critical Segments", len(critical_df))
        
        with col_exp2:
            st.markdown("#### 📊 Full Analysis Report")
            
            # Generate comprehensive report
            report = io.StringIO()
            report.write("=" * 60 + "\n")
            report.write("PALOCH FIELD GUARDIAN (PFG AI) - ANALYSIS REPORT\n")
            report.write("=" * 60 + "\n\n")
            report.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            report.write("EXECUTIVE SUMMARY\n")
            report.write("-" * 60 + "\n")
            report.write(f"Total Segments Analyzed: {len(df):,}\n")
            report.write(f"Critical Risk Segments: {critical_count:,} ({critical_count/len(df)*100:.1f}%)\n")
            report.write(f"Medium Risk Segments: {medium_count:,} ({medium_count/len(df)*100:.1f}%)\n")
            report.write(f"Safe Segments: {safe_count:,} ({safe_count/len(df)*100:.1f}%)\n")
            report.write(f"Maximum Leak Probability: {df['predicted_leak_prob'].max():.3f}\n")
            report.write(f"Average Leak Probability: {df['predicted_leak_prob'].mean():.3f}\n\n")
            
            if metrics:
                report.write("MODEL PERFORMANCE\n")
                report.write("-" * 60 + "\n")
                if metrics.get('auc'):
                    report.write(f"ROC AUC Score: {metrics['auc']:.3f}\n")
                if metrics.get('accuracy'):
                    report.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
                report.write("\n")
            
            report.write("TOP 20 CRITICAL SEGMENTS\n")
            report.write("-" * 60 + "\n")
            top_20 = df.sort_values('predicted_leak_prob', ascending=False).head(20)
            report.write(top_20[export_cols[:5]].to_string(index=False))
            report.write("\n\n")
            
            report.write("FEATURE IMPORTANCE (TOP 10)\n")
            report.write("-" * 60 + "\n")
            if feature_importance:
                for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                    report.write(f"{feat:.<40} {imp:.4f}\n")
            
            report.write("\n" + "=" * 60 + "\n")
            report.write("End of Report\n")
            
            st.download_button(
                label="📑 Download Full Report (TXT)",
                data=report.getvalue(),
                file_name=f"pfg_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Full dataset export
        st.markdown("---")
        st.markdown("#### 📦 Complete Dataset with Predictions")
        full_csv = df.to_csv(index=False)
        st.download_button(
            label=f"📊 Download Full Dataset ({len(df):,} rows)",
            data=full_csv,
            file_name=f"pfg_full_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Paloch Field Guardian (PFG AI)</strong> - Production Ready v1.0</p>
        <p style='font-size: 0.9em;'>
            🔒 For production deployment, integrate with live SCADA/sensor feeds and calibrate model with field data.<br>
            ⚠️ Predictions are based on historical patterns - always verify with domain experts.
        </p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)
        st.info("Please refresh the page or contact support if the issue persists.")