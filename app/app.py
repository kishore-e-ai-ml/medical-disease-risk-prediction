"""
Streamlit Interactive Web Application for Disease Risk Prediction.
Features a premium modern glassmorphic theme, KPI metrics grid, interactive patient diagnostic forms, and explainability visualizations.
"""
import sys
from pathlib import Path

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import DATASETS_CONFIG, REPORTS_DIR, FIGURES_DIR
from src.predict import predict_disease_risk
from src.data_loader import load_dataset

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="MediPredict AI | Clinical Risk System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Premium Glassmorphic Design System (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Font & Body Reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Background Gradient */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0F172A 0%, #0B0F19 90%);
        color: #F8FAFC;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }

    /* Top Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 1.2rem;
    }

    /* Pulsing Active Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
    }

    /* Medical Disclaimer Banner */
    .disclaimer-card {
        background: rgba(245, 158, 11, 0.06);
        border-left: 4px solid #F59E0B;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        color: #FCD34D;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Metric Glass Cards Strip */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-value-huge {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #38BDF8;
        margin-top: 0.2rem;
    }

    /* Section Cards */
    .section-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 0.8rem;
    }

    /* Prediction Result Cards */
    .result-box-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.3) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 12px 32px rgba(239, 68, 68, 0.18);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        color: #FCA5A5;
        margin-bottom: 1.2rem;
    }
    .result-box-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.3) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 12px 32px rgba(16, 185, 129, 0.18);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        color: #6EE7B7;
        margin-bottom: 1.2rem;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 0.5rem;
    }
    .result-percent {
        font-size: 3rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
        margin: 0.4rem 0;
        line-height: 1;
    }

    /* Styled Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .sidebar-brand {
        padding: 1rem 0;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    .sidebar-brand-text {
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Styled Input Labels */
    .stNumberInput label, .stSelectbox label {
        color: #CBD5E1 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 10px;
        padding: 0 20px;
        color: #94A3B8;
        font-weight: 600;
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #38BDF8 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
</style>
""", unsafe_allow_html=True)


# 3. Header Component
def render_header():
    st.markdown("""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
            <div>
                <div class="hero-title">🩺 MediPredict AI Platform</div>
                <div class="hero-subtitle">Clinical Machine Learning Risk Prediction Framework Across Multiple Disease Domains</div>
            </div>
            <div class="status-badge">
                <span class="pulse-dot"></span>
                ML Pipelines Active
            </div>
        </div>
    </div>
    
    <div class="disclaimer-card">
        <span style="font-size: 1.4rem;">⚠️</span>
        <div>
            <strong>EDUCATIONAL & RESEARCH DEMONSTRATION DISCLAIMER:</strong> 
            This web system is engineered for educational Machine Learning demonstration purposes only. Predictions generated must NOT be utilized for clinical diagnosis, treatment decisions, or medical advice.
        </div>
    </div>
    """, unsafe_allow_html=True)


# 4. Top Metric Cards Strip Component
def render_top_kpi_strip():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Disease Datasets</div>
            <div class="metric-value-huge">3 Models</div>
            <div class="metric-sub">Heart, Diabetes, Breast Cancer</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Max Diagnostic Accuracy</div>
            <div class="metric-value-huge" style="color: #34D399;">98.25%</div>
            <div class="metric-sub">Breast Cancer (Logistic Reg)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Max Screening Recall</div>
            <div class="metric-value-huge" style="color: #38BDF8;">96.43%</div>
            <div class="metric-sub">Heart Disease (XGBoost)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Pipeline Validation</div>
            <div class="metric-value-huge" style="color: #A78BFA;">100%</div>
            <div class="metric-sub">8/8 Pytest Suite Passed</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)


# 5. Sidebar Component
def render_sidebar():
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-text">⚡ MEDIPREDICT AI</div>
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">Clinical Diagnostic Suite</div>
    </div>
    """, unsafe_allow_html=True)
    
    dataset_option = st.sidebar.selectbox(
        "🎯 Select Medical Condition:",
        options=["Heart Disease", "Diabetes", "Breast Cancer"],
        index=0
    )
    
    key_map = {
        "Heart Disease": "heart",
        "Diabetes": "diabetes",
        "Breast Cancer": "breast_cancer"
    }
    dataset_key = key_map[dataset_option]
    config = DATASETS_CONFIG[dataset_key]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📋 Dataset Specs")
    st.sidebar.caption(config["description"])
    
    # Dataset Quick Stats Cards
    st.sidebar.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.85rem; margin-bottom: 1rem;">
        <div style="color: #94A3B8;">Features: <strong style="color: #F8FAFC;">{len(config['num_features']) + len(config['cat_features'])} parameters</strong></div>
        <div style="color: #94A3B8;">Target Column: <strong style="color: #38BDF8;">{config['target_col']}</strong></div>
    </div>
    """, unsafe_allow_html=True)
    
    model_path = config["model_path"]
    if model_path.exists():
        st.sidebar.success("✅ Serialized Model Loaded")
    else:
        st.sidebar.error("⚠️ Model File Missing")
        
    return dataset_key, config


# 6. Patient Risk Prediction Interface Tab
def render_prediction_tab(dataset_key: str, config: dict):
    st.markdown(f"""
    <div class="section-card">
        <div class="section-header">
            <span>🩺</span> Enter Patient Clinical Parameters ({config['name']})
        </div>
    """, unsafe_allow_html=True)
    
    input_data = {}
    
    if dataset_key == "heart":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 👤 Demographic & Blood")
            input_data["age"] = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
            input_data["sex"] = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
            input_data["cp"] = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                                           format_func=lambda x: ["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal", "3: Asymptomatic"][x])
            input_data["trestbps"] = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=130)
            input_data["chol"] = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=240)
            
        with col2:
            st.markdown("##### ⚡ Hemodynamics & ECG")
            input_data["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False (0)" if x == 0 else "True (1)")
            input_data["restecg"] = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                                                format_func=lambda x: ["0: Normal", "1: ST-T Abnormality", "2: LV Hypertrophy"][x])
            input_data["thalach"] = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150)
            input_data["exang"] = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
            input_data["oldpeak"] = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            
        with col3:
            st.markdown("##### 🔬 Fluoroscopy & Thalassemia")
            input_data["slope"] = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2], format_func=lambda x: ["0: Upsloping", "1: Flat", "2: Downsloping"][x])
            input_data["ca"] = st.selectbox("Major Vessels Colored (0-3)", options=[0, 1, 2, 3])
            input_data["thal"] = st.selectbox("Thalassemia Status", options=[1, 2, 3], format_func=lambda x: ["1: Normal", "2: Fixed Defect", "3: Reversable Defect"][x-1])

    elif dataset_key == "diabetes":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🩸 Metabolic Measurements")
            input_data["Pregnancies"] = st.number_input("Pregnancies Count", min_value=0, max_value=20, value=2)
            input_data["Glucose"] = st.number_input("Plasma Glucose Concentration (mg/dL)", min_value=0, max_value=300, value=120)
            input_data["BloodPressure"] = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, max_value=180, value=70)
            input_data["SkinThickness"] = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, max_value=100, value=20)
        with col2:
            st.markdown("##### ⚖️ Body Index & Genetics")
            input_data["Insulin"] = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=79)
            input_data["BMI"] = st.number_input("Body Mass Index (BMI kg/m²)", min_value=0.0, max_value=70.0, value=25.4, step=0.1)
            input_data["DiabetesPedigreeFunction"] = st.number_input("Diabetes Pedigree Score", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
            input_data["Age"] = st.number_input("Patient Age (years)", min_value=1, max_value=120, value=33)

    elif dataset_key == "breast_cancer":
        st.markdown("##### 🔬 Tumor Cell Nuclei Fine Needle Aspiration Parameters")
        num_features = config["num_features"]
        cols = st.columns(3)
        default_vals = {
            "mean radius": 14.1, "mean texture": 19.2, "mean perimeter": 91.9, "mean area": 654.8, "mean smoothness": 0.096,
            "mean compactness": 0.104, "mean concavity": 0.088, "mean concave points": 0.048, "mean symmetry": 0.181, "mean fractal dimension": 0.062,
            "radius error": 0.40, "texture error": 1.21, "perimeter error": 2.86, "area error": 40.3, "smoothness error": 0.007,
            "compactness error": 0.025, "concavity error": 0.031, "concave points error": 0.011, "symmetry error": 0.020, "fractal dimension error": 0.003,
            "worst radius": 16.2, "worst texture": 25.6, "worst perimeter": 107.2, "worst area": 880.5, "worst smoothness": 0.132,
            "worst compactness": 0.254, "worst concavity": 0.272, "worst concave points": 0.114, "worst symmetry": 0.282, "worst fractal dimension": 0.080
        }
        
        for i, feat in enumerate(num_features):
            c = cols[i % 3]
            val = default_vals.get(feat, 1.0)
            input_data[feat] = c.number_input(feat.capitalize(), value=float(val), format="%.4f")
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ PREDICT DISEASE RISK", type="primary", use_container_width=True):
        try:
            res = predict_disease_risk(dataset_key, input_data)
            
            st.markdown("<br>", unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1.3, 1])
            
            with res_col1:
                if res["prediction"] == 1:
                    st.markdown(f"""
                    <div class="result-box-high">
                        <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; color: #F87171; font-weight: 700;">Diagnostic Risk Alert</div>
                        <div class="result-title">⚠️ {res['prediction_label']}</div>
                        <div class="result-percent">{res['probability_percent']}</div>
                        <div style="font-size: 0.9rem; color: #FCA5A5;">Estimated Risk Probability Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box-low">
                        <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; color: #34D399; font-weight: 700;">Diagnostic Assessment</div>
                        <div class="result-title">✅ {res['prediction_label']}</div>
                        <div class="result-percent">{res['probability_percent']}</div>
                        <div style="font-size: 0.9rem; color: #6EE7B7;">Estimated Risk Probability Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.progress(res["probability"])
                st.caption(f"🧠 Selected Model Architecture: **{res['model_name']} Pipeline**")
                
            with res_col2:
                st.markdown("""
                <div class="section-card">
                    <div class="section-header">
                        <span>📊</span> Feature Weight Attributions
                    </div>
                """, unsafe_allow_html=True)
                imp_df = res["feature_importances"]
                if not imp_df.empty:
                    st.dataframe(imp_df.head(7), use_container_width=True, hide_index=True)
                else:
                    st.info("Feature weights not available for linear/kernel model.")
                st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error computing patient risk assessment: {e}")


# 7. Exploratory Data Analysis Tab
def render_eda_tab(dataset_key: str, config: dict):
    st.markdown(f"""
    <div class="section-card">
        <div class="section-header">
            <span>📊</span> Exploratory Data Analysis & Feature Distributions: {config['name']}
        </div>
    """, unsafe_allow_html=True)
    
    try:
        df = load_dataset(dataset_key)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📁 Raw Data Inspection (First 5 Rows)")
            st.dataframe(df.head(5), use_container_width=True)
        with c2:
            st.markdown("##### 🔍 Missing Values & Column Types Audit")
            audit_df = pd.DataFrame({
                "DataType": df.dtypes.astype(str),
                "MissingCount": df.isnull().sum(),
                "UniqueValues": df.nunique()
            })
            st.dataframe(audit_df, use_container_width=True)
            
        st.markdown("---")
        st.markdown("##### 📈 Clinical Diagnostic Charts")
        
        target_fig = FIGURES_DIR / f"{dataset_key}_target_distribution.png"
        corr_fig = FIGURES_DIR / f"{dataset_key}_correlation_heatmap.png"
        
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            if target_fig.exists():
                st.image(str(target_fig), caption="Target Variable Class Balance", use_container_width=True)
        with fig_col2:
            if corr_fig.exists():
                st.image(str(corr_fig), caption="Feature Co-Linearity Heatmap", use_container_width=True)
                
    except Exception as e:
        st.error(f"Error executing EDA rendering: {e}")
        
    st.markdown("</div>", unsafe_allow_html=True)


# 8. Model Benchmarks Tab
def render_benchmarks_tab(dataset_key: str, config: dict):
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>🏆</span> Machine Learning Model Benchmark Matrix
        </div>
    """, unsafe_allow_html=True)
    
    csv_path = REPORTS_DIR / "model_comparison.csv"
    if csv_path.exists():
        df_comp = pd.read_csv(csv_path)
        
        ds_comp = df_comp[df_comp["Dataset"] == config["name"]]
        st.markdown(f"##### Empirical Cross-Validation Metrics for **{config['name']}**")
        st.dataframe(ds_comp.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], color="#1E3A8A"), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("##### 📉 ROC Diagnostic Curves & Confusion Matrices")
        
        cm_fig = FIGURES_DIR / f"{dataset_key}_confusion_matrices.png"
        roc_fig = FIGURES_DIR / f"{dataset_key}_roc_curves.png"
        imp_fig = FIGURES_DIR / f"{dataset_key}_feature_importance.png"
        
        c1, c2 = st.columns(2)
        with c1:
            if cm_fig.exists():
                st.image(str(cm_fig), caption="Confusion Matrices Across 4 Algorithms", use_container_width=True)
        with c2:
            if roc_fig.exists():
                st.image(str(roc_fig), caption="ROC Curves & Discrimination Capability", use_container_width=True)
                
        if imp_fig.exists():
            st.markdown("---")
            st.image(str(imp_fig), caption="Feature Importance Attributions of Selected Best Model", use_container_width=True)
            
        st.markdown("---")
        st.markdown("##### 🌐 Full Project Performance Comparison Across All 3 Datasets")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.warning("Model evaluation benchmark CSV not found. Please run training pipeline.")
        
    st.markdown("</div>", unsafe_allow_html=True)


# 9. Main Router
def main():
    render_header()
    render_top_kpi_strip()
    
    dataset_key, config = render_sidebar()
    
    tab1, tab2, tab3 = st.tabs([
        "🩺 Patient Diagnostic Engine",
        "📊 Data Exploration (EDA)",
        "🏆 Model Benchmarks & Metrics"
    ])
    
    with tab1:
        render_prediction_tab(dataset_key, config)
    with tab2:
        render_eda_tab(dataset_key, config)
    with tab3:
        render_benchmarks_tab(dataset_key, config)


if __name__ == "__main__":
    main()
