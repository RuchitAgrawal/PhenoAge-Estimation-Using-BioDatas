"""
Biological Age & Longevity Estimation: Interactive Clinical Research Application
Built with Streamlit and Scikit-Learn / XGBoost.
Academic Research Prototype - Strictly for Educational and Research Purposes.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Biological Age & Longevity Estimator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value-primary {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2b6cb0;
    }
    .metric-value-accelerated {
        font-size: 2.5rem;
        font-weight: 800;
        color: #c53030;
    }
    .metric-value-decelerated {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2f855a;
    }
    .metric-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .disclaimer-box {
        background-color: #fffaf0;
        border-left: 4px solid #dd6b20;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 25px;
        margin-bottom: 25px;
        font-size: 0.92rem;
        color: #7b341e;
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(BASE_DIR, "results", "models", "scaler_A.joblib")):
    candidate = os.path.join(BASE_DIR, "biological-age-ml")
    if os.path.exists(os.path.join(candidate, "results", "models", "scaler_A.joblib")):
        BASE_DIR = candidate

MODELS_DIR = os.path.join(BASE_DIR, "results/models/exp_A")
SCALER_PATH = os.path.join(BASE_DIR, "results/models/scaler_A.joblib")

@st.cache_resource(show_spinner="Loading pre-trained model weights & clinical scaler...")
def load_models_and_scaler():
    """Load pre-trained models and standardization scaler with graceful fallbacks."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    
    candidates = [
        ("Neural Network (MLP - Best Performer)", "neural_network.joblib"),
        ("XGBoost Regressor", "xgboost.joblib"),
        ("Gradient Boosting Regressor", "gradient_boosting.joblib"),
        ("Random Forest Regressor", "random_forest.joblib"),
        ("Linear Regression (Baseline)", "linear_regression.joblib"),
        ("Support Vector Regressor (SVR)", "svr.joblib")
    ]
    
    models = {}
    for display_name, fname in candidates:
        m_path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(m_path):
            try:
                models[display_name] = joblib.load(m_path)
            except Exception:
                pass
                
    if not models:
        raise FileNotFoundError(f"No valid model artifacts discovered in {MODELS_DIR}")
    return models, scaler

try:
    available_models, scaler = load_models_and_scaler()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure `run_pipeline.py` has been executed.")
    st.stop()

# Header
st.markdown('<div class="main-header">Biological Age & Longevity Estimation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Routine Biomarker Multi-Model Inference & Survival Risk Stratification (NHANES IV Cohort)</div>', unsafe_allow_html=True)

# Sidebar - Settings & Preset Profiles
with st.sidebar:
    st.header("⚙️ Model & Presets")
    selected_model_name = st.selectbox(
        "Inference Model",
        options=list(available_models.keys()),
        index=0,
        help="Select among the 6 trained ML algorithms evaluated in the research study."
    )
    
    st.markdown("---")
    st.subheader("📋 Load Benchmark Profiles")
    preset = st.radio(
        "Populate Inputs With:",
        ["Cohort Average (Mean)", "Accelerated Aging Profile", "Decelerated Aging Profile", "Custom"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🔬 About This Study")
    st.info(
        "Based on the **BioAge** repository & NHANES IV benchmark cohort (Dayoon Kwon et al.). "
        "Trained on **35,723** adult participants across 13 clinical biomarkers to model PhenoAge biological aging."
    )

# Establish Default Values according to preset
if preset == "Cohort Average (Mean)":
    init_age = 50.0
    init_alb = 42.3
    init_alp = 72.0
    init_crp = 1.33 # lncrp ~ 0.285
    init_chol = 197.0
    init_creat = 76.0 # lncreat ~ 4.33 umol/L
    init_hba1c = 5.6
    init_sbp = 125.0
    init_bun = 13.6
    init_uap = 5.4
    init_lymph = 30.3
    init_mcv = 89.5
    init_wbc = 7.2
elif preset == "Accelerated Aging Profile":
    init_age = 50.0
    init_alb = 36.0 # Lower albumin
    init_alp = 95.0 # Higher ALP
    init_crp = 3.5  # Higher CRP
    init_chol = 240.0 # Higher cholesterol
    init_creat = 105.0 # Higher creatinine
    init_hba1c = 6.8  # Elevated glycated hemoglobin
    init_sbp = 145.0 # Elevated SBP
    init_bun = 19.0 # Elevated BUN
    init_uap = 6.8 # Elevated uric acid
    init_lymph = 22.0 # Lower lymphocytes
    init_mcv = 95.0
    init_wbc = 9.8 # Elevated WBC
elif preset == "Decelerated Aging Profile":
    init_age = 50.0
    init_alb = 46.0 # High albumin
    init_alp = 55.0 # Low-normal ALP
    init_crp = 0.5  # Low CRP
    init_chol = 180.0
    init_creat = 65.0 # Optimal creatinine
    init_hba1c = 5.1  # Optimal glycemic control
    init_sbp = 112.0 # Optimal BP
    init_bun = 11.0
    init_uap = 4.5
    init_lymph = 35.0
    init_mcv = 87.0
    init_wbc = 5.5  # Low-normal WBC
else:
    init_age, init_alb, init_alp, init_crp, init_chol, init_creat, init_hba1c, init_sbp, init_bun, init_uap, init_lymph, init_mcv, init_wbc = (
        50.0, 42.0, 70.0, 1.2, 195.0, 75.0, 5.5, 120.0, 13.0, 5.2, 30.0, 89.0, 7.0
    )

# Section 1: Biological Age Estimator Form
st.subheader("1. Clinical Biomarker Inputs")

with st.form("biomarker_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_in = st.number_input("Chronological Age (Years)", min_value=20.0, max_value=85.0, value=float(init_age), step=1.0)
        alb_in = st.number_input("Albumin (g/L)", min_value=20.0, max_value=60.0, value=float(init_alb), step=0.5, help="Liver function & nutritional status")
        alp_in = st.number_input("Alkaline Phosphatase - ALP (U/L)", min_value=10.0, max_value=220.0, value=float(init_alp), step=1.0, help="Liver and bone turnover")
        crp_in = st.number_input("C-Reactive Protein - CRP (mg/dL)", min_value=0.01, max_value=6.0, value=float(init_crp), step=0.1, help="Systemic inflammation")
        
    with col2:
        chol_in = st.number_input("Total Cholesterol (mg/dL)", min_value=70.0, max_value=410.0, value=float(init_chol), step=1.0)
        creat_in = st.number_input("Creatinine (umol/L)", min_value=20.0, max_value=300.0, value=float(init_creat), step=1.0, help="Renal filtration function")
        hba1c_in = st.number_input("Glycated Hemoglobin - HbA1c (%)", min_value=3.0, max_value=12.0, value=float(init_hba1c), step=0.1, help="Long-term glucose homeostasis")
        sbp_in = st.number_input("Systolic Blood Pressure - SBP (mmHg)", min_value=70.0, max_value=230.0, value=float(init_sbp), step=1.0)

    with col3:
        bun_in = st.number_input("Blood Urea Nitrogen - BUN (mg/dL)", min_value=1.0, max_value=50.0, value=float(init_bun), step=0.5)
        uap_in = st.number_input("Uric Acid (mg/dL)", min_value=0.5, max_value=13.0, value=float(init_uap), step=0.1)
        lymph_in = st.number_input("Lymphocyte Percentage (%)", min_value=3.0, max_value=75.0, value=float(init_lymph), step=0.5, help="Adaptive immunity")
        mcv_in = st.number_input("Mean Corpuscular Volume - MCV (fL)", min_value=55.0, max_value=120.0, value=float(init_mcv), step=0.5)
        wbc_in = st.number_input("White Blood Cell Count - WBC (10^3/uL)", min_value=1.0, max_value=20.0, value=float(init_wbc), step=0.1, help="Innate immune activity")

    submit_button = st.form_submit_button("Calculate Biological Age & Survival Risk", use_container_width=True)

# Feature transformation
lncrp_val = np.log(max(crp_in, 0.001))
lncreat_val = np.log(max(creat_in, 1.0))

features_raw = np.array([[
    age_in, alb_in, alp_in, lncrp_val, chol_in,
    lncreat_val, hba1c_in, sbp_in, bun_in, uap_in,
    lymph_in, mcv_in, wbc_in
]])

# Preprocessing: Scale using the training-fitted scaler
features_scaled = scaler.transform(features_raw)

# Inference
model = available_models[selected_model_name]
pred_bio_age = float(model.predict(features_scaled)[0])
bio_acceleration = pred_bio_age - age_in

# Root Mean Square Error of Best Model for 95% Confidence Interval
test_rmse = 3.8643 # Empirical test RMSE from Experiment A
ci_lower = pred_bio_age - 1.96 * test_rmse
ci_upper = pred_bio_age + 1.96 * test_rmse

# Section 2: Results Cards
st.markdown("---")
st.subheader("2. Biological Age Estimation Results")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Chronological Age</div>
        <div class="metric-value-primary">{age_in:.1f}</div>
        <div style="color: #718096; font-size: 0.9rem; margin-top: 6px;">Calendar Years Lived</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Biological Age (PhenoAge)</div>
        <div class="metric-value-primary">{pred_bio_age:.1f}</div>
        <div style="color: #718096; font-size: 0.9rem; margin-top: 6px;">95% Pred. Interval: [{ci_lower:.1f}, {ci_upper:.1f}] Years</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    if bio_acceleration > 0.5:
        accel_class = "metric-value-accelerated"
        trend_text = f"+{bio_acceleration:.1f} Years (Accelerated Aging)"
        trend_sub = "Higher physiological aging rate than chronological age"
    elif bio_acceleration < -0.5:
        accel_class = "metric-value-decelerated"
        trend_text = f"{bio_acceleration:.1f} Years (Decelerated Aging)"
        trend_sub = "Lower physiological aging rate than chronological age"
    else:
        accel_class = "metric-value-primary"
        trend_text = f"{bio_acceleration:+.1f} Years (Age-Congruent)"
        trend_sub = "Biological age matches chronological expectation"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Biological Age Acceleration</div>
        <div class="{accel_class}">{trend_text.split()[0]}</div>
        <div style="font-weight: 600; color: #4a5568; margin-top: 4px;">{trend_text.split('(', 1)[-1].replace(')', '') if '(' in trend_text else ''}</div>
        <div style="color: #718096; font-size: 0.85rem; margin-top: 2px;">{trend_sub}</div>
    </div>
    """, unsafe_allow_html=True)

# Section 3: Longevity & Survival Risk Stratification
st.markdown("---")
st.subheader("3. Mortality Risk & Survival Association (Empirical Cox Model)")

# Using the empirical Cox PH multivariable model:
# HR per year of acceleration = 1.0678 (95% CI: 1.0516 - 1.0842)
hr_base = 1.0678
hr_low_base = 1.0516
hr_high_base = 1.0842

rel_hr = hr_base ** bio_acceleration
rel_hr_low = hr_low_base ** bio_acceleration if bio_acceleration >= 0 else hr_high_base ** bio_acceleration
rel_hr_high = hr_high_base ** bio_acceleration if bio_acceleration >= 0 else hr_low_base ** bio_acceleration

surv_col1, surv_col2 = st.columns([1, 1])

with surv_col1:
    st.markdown("##### 📈 Relative Mortality Hazard")
    st.write(
        f"Based on multivariable Cox proportional hazards regression adjusted for chronological age "
        f"on the NHANES follow-up cohort (N=5,280 holdout survival sample, median follow-up 116 months):"
    )
    st.metric(
        label="Relative Hazard Ratio (vs. Age-Matched Baseline)",
        value=f"{rel_hr:.2f}x",
        delta=f"{(rel_hr - 1.0)*100:+.1f}% Risk" if abs(rel_hr - 1.0) > 0.02 else "Baseline Risk",
        delta_color="inverse"
    )
    st.caption(f"Estimated 95% Confidence Interval: [{min(rel_hr_low, rel_hr_high):.2f}x – {max(rel_hr_low, rel_hr_high):.2f}x] (p < 0.0001)")

with surv_col2:
    st.markdown("##### 🔬 Risk Stratification Interpretation")
    if bio_acceleration > 2.0:
        st.warning(
            f"**Statistically Accelerated Profile (+{bio_acceleration:.1f} yrs):** In NHANES epidemiological data, individuals "
            f"with accelerated biological age exhibit significantly higher all-cause mortality hazard (log-rank p < 0.0001). "
            f"Key driving indicators commonly include elevated HbA1c, inflammatory markers (CRP, WBC), or subclinical renal/metabolic strain."
        )
    elif bio_acceleration < -2.0:
        st.success(
            f"**Statistically Decelerated Profile ({bio_acceleration:.1f} yrs):** Biological age indicators are favorable compared "
            f"to chronological cohort peers. In NHANES survival cohorts, decelerated biological age is associated with "
            f"prolonged event-free survival and lower relative hazard of all-cause mortality."
        )
    else:
        st.info(
            f"**Baseline Congruent Profile ({bio_acceleration:+.1f} yrs):** Biological age closely aligns with chronological expectations. "
            f"Participant biomarkers reflect average population physiological aging rates."
        )

# Section 4: Biomarker Analysis & Feature Importance
st.markdown("---")
st.subheader("4. Biomarker Analysis & Contributing Factors")

feat_col1, feat_col2 = st.columns([1, 1])

with feat_col1:
    st.markdown("##### 📊 Population Feature Importance (SHAP Ranking)")
    shap_path = os.path.join(BASE_DIR, "results/tables/shap_importance_ranking.csv")
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path)
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
        plot_df = shap_df[shap_df["Feature"] != "age"].head(8)
        ax.barh(plot_df["DisplayName"], plot_df["Mean_Abs_SHAP"], color="#3182ce", edgecolor="black")
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP Value| (Impact on BioAge in Years)", fontsize=9)
        ax.set_title("Top Biomarkers Influencing PhenoAge", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with feat_col2:
    st.markdown("##### 🩸 Key Clinical Biomarker Mechanisms")
    st.markdown("""
    - **Glycated Hemoglobin (HbA1c):** Reflects 90-day glycemic control; advanced glycation end-products drive vascular stiffening and systemic senescence.
    - **Creatinine (lncreat):** Indicator of renal filtration efficiency and muscle mass integrity.
    - **Albumin (albumin_gL):** Major serum protein; declining levels indicate systemic micro-inflammation, hepatic decline, and frailty.
    - **Lymphocyte Percentage:** Hallmark of immunosenescence; adaptive immune contraction with age.
    - **WBC & C-Reactive Protein (CRP):** Indicators of sterile chronic low-grade systemic inflammation ('inflammaging').
    - **Alkaline Phosphatase (ALP):** Biomarker of hepatobiliary function and bone turnover.
    """)

# Section 5: Academic Research Information & Disclaimer
st.markdown("---")
st.subheader("5. Research Architecture & Methodology")

with st.expander("📚 Model Architecture, Cross-Validation & Dataset Details"):
    st.markdown("""
    - **Dataset:** NHANES IV (1999–2018) via the R `BioAge` package by Dayoon Kwon et al.
    - **Complete Cohort:** 35,723 adult participants across 14 validated variables.
    - **Leakage Prevention:** 80/20 train/test holdout split (fixed random seed = 42). All scalers fitted strictly on the training partition.
    - **Algorithms Evaluated:** Linear Regression (Baseline), Random Forest, Linear SVR, Gradient Boosting, XGBoost, and Multilayer Perceptron (Neural Network).
    - **Validation Protocol:** 5-fold cross-validation on training data, followed by final untouched evaluation on holdout test set (N=7,145).
    - **Survival Methodology:** Kaplan-Meier survival curves, log-rank significance test, and Cox Proportional Hazards regression (univariate and multivariable age-adjusted).
    """)

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ ACADEMIC RESEARCH AND EDUCATIONAL DISCLAIMER:</strong><br>
    This project is strictly an academic research and educational demonstration developed for a Machine Learning curriculum. 
    Biological age estimates and survival associations represent observational statistical predictions derived from population-level 
    epidemiological data (NHANES IV) and <strong>must NOT be interpreted as medical diagnoses, lifespan predictions, or individual medical advice</strong>.
    Consult qualified medical professionals for clinical evaluations and health decision-making.
</div>
""", unsafe_allow_html=True)
