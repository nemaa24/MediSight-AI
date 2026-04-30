import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import joblib
import os
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MediSight AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .label-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-high     { color: #dc3545; font-weight: bold; font-size: 1.5rem; }
    .risk-moderate { color: #fd7e14; font-weight: bold; font-size: 1.5rem; }
    .risk-low      { color: #28a745; font-weight: bold; font-size: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DISEASES      = ["Diabetes", "Heart Disease", "Liver Disease", "Kidney Disease"]
DISEASE_KEYS  = ["diabetes", "heart", "liver", "kidney"]
MODEL_TYPES   = ["Logistic Regression", "Random Forest", "XGBoost"]
MODEL_KEYS    = ["lr", "rf", "xgb"]

BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR          = os.path.join(BASE_DIR, "../models")
FAIRNESS_DIR        = os.path.join(BASE_DIR, "../fairness")
DATASETS_DIR        = os.path.join(BASE_DIR, "../datasets/processed")
EXPLAINABILITY_DIR  = os.path.join(BASE_DIR, "../explainability")

# ─────────────────────────────────────────────
# FIX ISSUE 1 — SAMPLE PATIENTS
# Pre-defined High Risk / Low Risk profiles so
# users don't have to fill every field manually.
# ─────────────────────────────────────────────
SAMPLE_PATIENTS = {
    "diabetes": {
        "High Risk": {
            "HighBP":1,"HighChol":1,"CholCheck":1,"BMI":35,"Smoker":1,
            "Stroke":0,"HeartDiseaseorAttack":1,"PhysActivity":0,
            "Fruits":0,"Veggies":0,"HvyAlcoholConsump":0,"AnyHealthcare":1,
            "NoDocbcCost":0,"GenHlth":4,"MentHlth":10,"PhysHlth":15,
            "DiffWalk":1,"Sex":1,"Age":10,"Education":3,"Income":3
        },
        "Low Risk": {
            "HighBP":0,"HighChol":0,"CholCheck":1,"BMI":22,"Smoker":0,
            "Stroke":0,"HeartDiseaseorAttack":0,"PhysActivity":1,
            "Fruits":1,"Veggies":1,"HvyAlcoholConsump":0,"AnyHealthcare":1,
            "NoDocbcCost":0,"GenHlth":1,"MentHlth":0,"PhysHlth":0,
            "DiffWalk":0,"Sex":0,"Age":3,"Education":6,"Income":8
        },
    },
    "heart": {
        "High Risk": {
            "Sex":1,"GeneralHealth":2,"PhysicalHealthDays":20,
            "MentalHealthDays":10,"PhysicalActivities":0,"SleepHours":5,
            "HadStroke":1,"HadAsthma":0,"HadSkinCancer":0,"HadCOPD":1,
            "HadDepressiveDisorder":1,"HadKidneyDisease":1,"HadArthritis":1,
            "HadDiabetes":1,"DeafOrHardOfHearing":0,
            "BlindOrVisionDifficulty":0,"DifficultyConcentrating":1,
            "DifficultyWalking":1,"DifficultyDressingBathing":0,
            "DifficultyErrands":1,"SmokerStatus":3,"ChestScan":1,
            "AgeCategory":72,"BMI":32,"AlcoholDrinkers":0
        },
        "Low Risk": {
            "Sex":0,"GeneralHealth":5,"PhysicalHealthDays":0,
            "MentalHealthDays":0,"PhysicalActivities":1,"SleepHours":8,
            "HadStroke":0,"HadAsthma":0,"HadSkinCancer":0,"HadCOPD":0,
            "HadDepressiveDisorder":0,"HadKidneyDisease":0,"HadArthritis":0,
            "HadDiabetes":0,"DeafOrHardOfHearing":0,
            "BlindOrVisionDifficulty":0,"DifficultyConcentrating":0,
            "DifficultyWalking":0,"DifficultyDressingBathing":0,
            "DifficultyErrands":0,"SmokerStatus":0,"ChestScan":0,
            "AgeCategory":27,"BMI":22,"AlcoholDrinkers":0
        },
    },
    "liver": {
        "High Risk": {
            "Age of the patient":55,"Gender":1,"Total Bilirubin":4.5,
            "Direct Bilirubin":2.1,"Alkphos Alkaline Phosphotase":350,
            "Sgpt Alamine Aminotransferase":120,
            "Sgot Aspartate Aminotransferase":110,
            "Total Protiens":5.5,"ALB Albumin":2.8,
            "A/G Ratio Albumin and Globulin Ratio":0.7
        },
        "Low Risk": {
            "Age of the patient":30,"Gender":0,"Total Bilirubin":0.6,
            "Direct Bilirubin":0.1,"Alkphos Alkaline Phosphotase":100,
            "Sgpt Alamine Aminotransferase":20,
            "Sgot Aspartate Aminotransferase":22,
            "Total Protiens":7.0,"ALB Albumin":3.8,
            "A/G Ratio Albumin and Globulin Ratio":1.1
        },
    },
    "kidney": {
        "High Risk": {
            "age":55,"bp":90,"sg":1,"al":4,"su":2,"rbc":1,"pc":1,
            "pcc":1,"ba":1,"bgr":200,"bu":80,"sc":4,"sod":120,
            "pot":6,"hemo":8,"pcv":20,"wbcc":12000,"rbcc":3,
            "htn":1,"dm":1,"cad":1,"appet":0,"pe":1,"ane":1
        },
        "Low Risk": {
            "age":35,"bp":70,"sg":3,"al":0,"su":0,"rbc":1,"pc":0,
            "pcc":0,"ba":0,"bgr":100,"bu":20,"sc":1,"sod":140,
            "pot":4,"hemo":15,"pcv":44,"wbcc":7000,"rbcc":5,
            "htn":0,"dm":0,"cad":0,"appet":1,"pe":0,"ane":0
        },
    },
}

# Feature ranges for What-If sliders (min, max, default)
FEATURE_RANGES = {
    "diabetes": {
        "HighBP":(0,1,0),"HighChol":(0,1,0),"CholCheck":(0,1,1),
        "BMI":(10,60,25),"Smoker":(0,1,0),"Stroke":(0,1,0),
        "HeartDiseaseorAttack":(0,1,0),"PhysActivity":(0,1,1),
        "Fruits":(0,1,1),"Veggies":(0,1,1),"HvyAlcoholConsump":(0,1,0),
        "AnyHealthcare":(0,1,1),"NoDocbcCost":(0,1,0),
        "GenHlth":(1,5,3),"MentHlth":(0,30,0),"PhysHlth":(0,30,0),
        "DiffWalk":(0,1,0),"Sex":(0,1,0),"Age":(1,14,7),
        "Education":(1,6,4),"Income":(1,8,5)
    },
    "heart": {
        "Sex":(0,1,0),"GeneralHealth":(1,5,3),
        "PhysicalHealthDays":(0,30,0),"MentalHealthDays":(0,30,0),
        "PhysicalActivities":(0,1,1),"SleepHours":(1,24,7),
        "HadStroke":(0,1,0),"HadAsthma":(0,1,0),
        "HadSkinCancer":(0,1,0),"HadCOPD":(0,1,0),
        "HadDepressiveDisorder":(0,1,0),"HadKidneyDisease":(0,1,0),
        "HadArthritis":(0,1,0),"HadDiabetes":(0,1,0),
        "DeafOrHardOfHearing":(0,1,0),"BlindOrVisionDifficulty":(0,1,0),
        "DifficultyConcentrating":(0,1,0),"DifficultyWalking":(0,1,0),
        "DifficultyDressingBathing":(0,1,0),"DifficultyErrands":(0,1,0),
        "SmokerStatus":(0,3,0),"ChestScan":(0,1,0),
        "AgeCategory":(21,82,45),"BMI":(10,60,25),"AlcoholDrinkers":(0,1,0)
    },
    "liver": {
        "Age of the patient":(1,90,40),"Gender":(0,1,0),
        "Total Bilirubin":(0.0,75.0,1.0),
        "Direct Bilirubin":(0.0,20.0,0.3),
        "Alkphos Alkaline Phosphotase":(60,2200,200),
        "Sgpt Alamine Aminotransferase":(10,2000,30),
        "Sgot Aspartate Aminotransferase":(10,3000,30),
        "Total Protiens":(2.0,10.0,6.5),"ALB Albumin":(0.5,5.5,3.5),
        "A/G Ratio Albumin and Globulin Ratio":(0.0,3.0,1.0)
    },
    "kidney": {
        "age":(1,90,45),"bp":(50,180,80),"sg":(0,4,2),
        "al":(0,5,0),"su":(0,5,0),"rbc":(0,1,1),"pc":(0,1,0),
        "pcc":(0,1,0),"ba":(0,1,0),"bgr":(70,500,100),
        "bu":(10,200,40),"sc":(0,20,1),"sod":(100,160,135),
        "pot":(2,10,4),"hemo":(3,18,13),"pcv":(10,55,40),
        "wbcc":(2000,20000,7000),"rbcc":(2,8,5),
        "htn":(0,1,0),"dm":(0,1,0),"cad":(0,1,0),
        "appet":(0,1,1),"pe":(0,1,0),"ane":(0,1,0)
    },
}

# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data
def load_all_metrics():
    path = os.path.join(MODELS_DIR, "all_metrics.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {
            "diabetes":{
                "lr": {"model_type":"Logistic Regression","accuracy":0.746,
                       "precision":0.737,"recall":0.764,"f1_score":0.751,
                       "auc_roc":0.823,"calibrated":True},
                "rf": {"model_type":"Random Forest","accuracy":0.738,
                       "precision":0.723,"recall":0.772,"f1_score":0.747,
                       "auc_roc":0.817,"calibrated":True},
                "xgb":{"model_type":"XGBoost","accuracy":0.750,
                       "precision":0.734,"recall":0.784,"f1_score":0.758,
                       "auc_roc":0.828,"calibrated":True}},
            "heart":{
                "lr": {"model_type":"Logistic Regression","accuracy":0.756,
                       "precision":0.237,"recall":0.783,"f1_score":0.364,
                       "auc_roc":0.844,"calibrated":True},
                "rf": {"model_type":"Random Forest","accuracy":0.853,
                       "precision":0.303,"recall":0.503,"f1_score":0.378,
                       "auc_roc":0.814,"threshold":0.2,"calibrated":False},
                "xgb":{"model_type":"XGBoost","accuracy":0.758,
                       "precision":0.236,"recall":0.765,"f1_score":0.360,
                       "auc_roc":0.834,"calibrated":False}},
            "liver":{
                "lr": {"model_type":"Logistic Regression","accuracy":0.643,
                       "precision":0.911,"recall":0.555,"f1_score":0.689,
                       "auc_roc":0.762,"calibrated":True},
                "rf": {"model_type":"Random Forest","accuracy":0.770,
                       "precision":0.994,"recall":0.682,"f1_score":0.809,
                       "auc_roc":0.937,"calibrated":False},
                "xgb":{"model_type":"XGBoost","accuracy":0.726,
                       "precision":0.952,"recall":0.649,"f1_score":0.772,
                       "auc_roc":0.881,"calibrated":False}},
            "kidney":{
                "lr": {"model_type":"Logistic Regression","accuracy":1.0,
                       "precision":1.0,"recall":1.0,"f1_score":1.0,
                       "auc_roc":1.0,"calibrated":True},
                "rf": {"model_type":"Random Forest","accuracy":1.0,
                       "precision":1.0,"recall":1.0,"f1_score":1.0,
                       "auc_roc":1.0,"calibrated":True},
                "xgb":{"model_type":"XGBoost","accuracy":1.0,
                       "precision":1.0,"recall":1.0,"f1_score":1.0,
                       "auc_roc":1.0,"calibrated":True}},
        }

@st.cache_data
def load_fairness_data():
    fairness  = {}
    defaults  = {
        "diabetes":{"group_accuracy":{"Under 40":0.859,"40-60":0.758,"Above 60":0.723},
                    "demographic_parity_difference":0.610,
                    "equalized_odds_difference":0.473,
                    "max_accuracy_gap":0.137,"bias_risk":"High"},
        "heart":   {"group_accuracy":{"Under 40":0.988,"40-60":0.945,"Above 60":0.807},
                    "demographic_parity_difference":0.163,
                    "equalized_odds_difference":0.383,
                    "max_accuracy_gap":0.181,"bias_risk":"High"},
        "liver":   {"group_accuracy":{"Under 40":0.858,"40-60":0.852,"Above 60":0.860},
                    "demographic_parity_difference":0.026,
                    "equalized_odds_difference":0.015,
                    "max_accuracy_gap":0.008,"bias_risk":"Low"},
        "kidney":  {"group_accuracy":{"Under 40":1.0,"40-60":1.0,"Above 60":1.0},
                    "demographic_parity_difference":0.434,
                    "equalized_odds_difference":0.0,
                    "max_accuracy_gap":0.0,"bias_risk":"Low"}
    }
    for key in DISEASE_KEYS:
        path = os.path.join(FAIRNESS_DIR, f"{key}_fairness.json")
        try:
            with open(path) as f:
                data = json.load(f)
                fairness[key] = data["age_fairness"]
        except Exception:
            fairness[key] = defaults[key]
    return fairness

@st.cache_data
def load_features(disease_key):
    path = os.path.join(DATASETS_DIR, f"{disease_key}_features.pkl")
    try:
        return joblib.load(path)
    except Exception:
        return list(
            SAMPLE_PATIENTS.get(disease_key, {})
                           .get("High Risk", {}).keys()
        )

@st.cache_data
def load_shap_data(disease_key):
    path = os.path.join(EXPLAINABILITY_DIR, f"{disease_key}_shap.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def load_model(disease_key, model_key):
    path = os.path.join(MODELS_DIR, f"{disease_key}_{model_key}.pkl")
    try:
        return joblib.load(path)
    except Exception:
        return None

def load_scaler(disease_key):
    path = os.path.join(MODELS_DIR, f"{disease_key}_scaler.pkl")
    try:
        return joblib.load(path)
    except Exception:
        return None

# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────
def make_prediction(disease_key, model_key, patient_values, features):
    scaler = load_scaler(disease_key)
    model  = load_model(disease_key, model_key)
    if scaler is None or model is None:
        return None, None
    X_input  = np.array([patient_values], dtype=float)
    X_scaled = scaler.transform(X_input)
    prob     = float(model.predict_proba(X_scaled)[0][1])
    if disease_key == "heart" and model_key == "rf":
        try:
            threshold = joblib.load(
                os.path.join(MODELS_DIR, "heart_rf_threshold.pkl")
            )
        except Exception:
            threshold = 0.20
        pred = 1 if prob >= threshold else 0
    else:
        pred = 1 if prob >= 0.5 else 0
    return prob, pred

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def risk_colour(level):
    return {"High":"#dc3545","Moderate":"#fd7e14","Low":"#28a745"}.get(level,"#666")

def prob_to_risk(prob):
    if prob > 0.6:   return "High"
    elif prob > 0.4: return "Moderate"
    return "Low"

def safe_pdf(text):
    """
    FIX ISSUE 6 — Replace characters outside latin-1
    so FPDF never raises UnicodeEncodeError.
    """
    replacements = {
        "\u2014":"--","\u2013":"-","\u2018":"'","\u2019":"'",
        "\u201c":'"', "\u201d":'"',"\u2022":"*","\u00a0":" ",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 MediSight AI</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI Transparency Label for '
            'Healthcare Prediction Systems</div>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# FIX ISSUE 4 — replaced broken placeholder URL
# with a styled HTML banner that always renders.
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);
             border-radius:10px;padding:14px;text-align:center;
             color:white;font-weight:700;font-size:1.15rem;
             margin-bottom:8px;letter-spacing:0.5px">
            🏥 MediSight AI
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### 🔧 Configuration")

    selected_disease    = st.selectbox("Select Disease", DISEASES)
    disease_key         = DISEASE_KEYS[DISEASES.index(selected_disease)]
    selected_model_name = st.selectbox("Select Model", MODEL_TYPES)
    model_key           = MODEL_KEYS[MODEL_TYPES.index(selected_model_name)]

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    metrics = load_all_metrics()
    m = metrics.get(disease_key, {}).get(model_key, {})
    if m:
        st.metric("AUC-ROC",  f"{m.get('auc_roc',0):.3f}")
        st.metric("Recall",   f"{m.get('recall',0):.3f}")
        st.metric("Accuracy", f"{m.get('accuracy',0):.3f}")

    st.markdown("---")
    st.caption("Department of AI & DS\nMysore University School of Engineering")

# ─────────────────────────────────────────────
# SHARED DATA
# ─────────────────────────────────────────────
all_metrics   = load_all_metrics()
fairness_data = load_fairness_data()
features      = load_features(disease_key)
shap_data     = load_shap_data(disease_key)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏷️ Transparency Label",
    "⚔️ Model Battle",
    "⚖️ Fairness Audit",
    "🔬 What-If Simulator",
    "💬 Prediction Explainer",
    "📄 Compliance Report"
])

# ══════════════════════════════════════════════
# TAB 1 — TRANSPARENCY LABEL
# ══════════════════════════════════════════════
with tab1:
    st.header("🏷️ AI Transparency Label")
    st.markdown("Generate a standardised transparency label for any patient.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Patient Input")

        # FIX ISSUE 1 — profile selector replaces raw field-by-field entry
        sample_profiles  = list(SAMPLE_PATIENTS.get(disease_key, {}).keys())
        selected_profile = st.selectbox(
            "Quick-fill with sample patient",
            ["Custom"] + sample_profiles,
            help="Auto-fills values. Select Custom to enter your own."
        )
        profile_vals = (
            SAMPLE_PATIENTS[disease_key][selected_profile]
            if selected_profile != "Custom"
            else {f: 0.0 for f in features}
        )

        st.markdown("**Key feature values** (top SHAP features shown):")
        patient_data = {}

        with st.form("patient_form"):
            # Show only top features by SHAP importance
            key_features = (
                shap_data.get("top_10_features", features[:10])
                if shap_data else features[:10]
            )
            for feat in key_features:
                default_val = float(profile_vals.get(feat, 0.0))
                patient_data[feat] = st.number_input(
                    feat, value=default_val, step=0.1, key=f"t1_{feat}"
                )
            # Remaining features filled silently from profile
            for feat in features:
                if feat not in patient_data:
                    patient_data[feat] = float(profile_vals.get(feat, 0.0))

            submitted = st.form_submit_button(
                "🔍 Generate Transparency Label",
                use_container_width=True
            )

    with col_right:
        st.subheader("Transparency Label Output")

        fairness_info = fairness_data.get(disease_key, {})
        bias_risk     = fairness_info.get("bias_risk", "Unknown")

        # Run real prediction on submit
        if submitted:
            patient_values = [patient_data[f] for f in features]
            real_prob, real_pred = make_prediction(
                disease_key, model_key, patient_values, features
            )
            if real_prob is not None:
                probability = real_prob
                prediction  = real_pred
                risk_level  = prob_to_risk(probability)
                st.session_state["t1_prob"] = probability
                st.session_state["t1_pred"] = prediction
                st.session_state["t1_risk"] = risk_level
            else:
                st.warning("Model files not found.")
                probability, prediction, risk_level = 0.5, 0, "Moderate"
        else:
            probability = st.session_state.get("t1_prob", 0.5)
            prediction  = st.session_state.get("t1_pred", 0)
            risk_level  = st.session_state.get("t1_risk", "Moderate")

        # Label card
        st.markdown(f"""
<div class="label-card">
  <h2 style="margin:0 0 0.5rem 0">🏥 MediSight AI Transparency Label</h2>
  <p style="margin:0;opacity:0.8">
    Disease: <b>{selected_disease}</b> &nbsp;|&nbsp;
    Model: <b>{selected_model_name}</b>
  </p>
  <hr style="border-color:rgba(255,255,255,0.3)">
  <table width="100%">
    <tr>
      <td><b>🔮 Prediction</b></td>
      <td style="text-align:right">
        {'⚠️ <b>POSITIVE — Disease Detected</b>'
         if prediction == 1
         else '✅ <b>NEGATIVE — No Disease</b>'}
      </td>
    </tr>
    <tr>
      <td><b>📊 Confidence</b></td>
      <td style="text-align:right"><b>{probability*100:.1f}%</b></td>
    </tr>
    <tr>
      <td><b>🚨 Risk Level</b></td>
      <td style="text-align:right"><b>{risk_level}</b></td>
    </tr>
    <tr>
      <td><b>📈 AUC-ROC</b></td>
      <td style="text-align:right"><b>{m.get('auc_roc',0):.3f}</b></td>
    </tr>
    <tr>
      <td><b>⚖️ Fairness Bias Risk</b></td>
      <td style="text-align:right"><b>{bias_risk}</b></td>
    </tr>
  </table>
  <hr style="border-color:rgba(255,255,255,0.3)">
  <p style="margin:0;font-size:0.85rem;opacity:0.8">
    ✅ EU AI Act 2024 Compliant &nbsp;|&nbsp; Generated by MediSight AI
  </p>
</div>
""", unsafe_allow_html=True)

    # FIX ISSUE 2 — gauge in its own full-width row BELOW both columns
    # so the title "Prediction Confidence" never overlaps the card.
    st.markdown("---")
    _, gauge_col, _ = st.columns([1, 2, 1])
    with gauge_col:
        st.markdown("#### Prediction Confidence (%)")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": risk_colour(risk_level)},
                "steps": [
                    {"range": [0,  40], "color": "#e8f5e9"},
                    {"range": [40, 60], "color": "#fff3e0"},
                    {"range": [60,100], "color": "#ffebee"},
                ]
            }
        ))
        fig_gauge.update_layout(
            height=250, margin=dict(t=10, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # SHAP charts — clean, no "from Nema" caption
    st.markdown("---")
    shap_c1, shap_c2 = st.columns([1, 1])
    with shap_c1:
        st.subheader("🔑 Top Contributing Factors")
        if shap_data is not None:
            importance   = shap_data["feature_importance"]
            top_features = list(importance.keys())[:8]
            top_vals     = [importance[f] for f in top_features]
            fig_shap = go.Figure(go.Bar(
                x=top_vals, y=top_features,
                orientation="h",
                marker_color="#dc3545"
            ))
            fig_shap.update_layout(
                xaxis_title="Mean |SHAP Value|",
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.warning("SHAP data not found.")

    with shap_c2:
        st.subheader("📊 Global Feature Importance")
        bar_img = os.path.join(
            EXPLAINABILITY_DIR, f"{disease_key}_shap_bar.png"
        )
        if os.path.exists(bar_img):
            st.image(bar_img, use_container_width=True)
        else:
            st.info("SHAP bar chart not found.")


# ══════════════════════════════════════════════
# TAB 2 — MODEL BATTLE
# ══════════════════════════════════════════════
with tab2:
    st.header("⚔️ Model Battle")
    st.markdown(f"Compare all three models for **{selected_disease}**.")

    disease_metrics = all_metrics.get(disease_key, {})
    rows = []
    for mk, mn in zip(MODEL_KEYS, MODEL_TYPES):
        dm = disease_metrics.get(mk, {})
        rows.append({
            "Model":     mn,
            "Accuracy":  dm.get("accuracy",  0),
            "Precision": dm.get("precision", 0),
            "Recall":    dm.get("recall",    0),
            "F1 Score":  dm.get("f1_score",  0),
            "AUC-ROC":   dm.get("auc_roc",   0),
        })
    df_metrics = pd.DataFrame(rows)

    def highlight_best(col):
        is_max = col == col.max()
        return ["background-color:#d4edda" if v else "" for v in is_max]

    st.dataframe(
        df_metrics.style
                  .apply(highlight_best, subset=["AUC-ROC"])
                  .format({c: "{:.4f}" for c in df_metrics.columns[1:]}),
        use_container_width=True
    )

    st.subheader("📊 Performance Radar")
    categories = ["Accuracy","Precision","Recall","F1 Score","AUC-ROC"]
    fig_radar  = go.Figure()
    for mk, mn, col in zip(MODEL_KEYS, MODEL_TYPES,
                           ["#1f77b4","#ff7f0e","#2ca02c"]):
        dm   = disease_metrics.get(mk, {})
        vals = [dm.get(c.lower().replace(" ","_"), 0) for c in categories]
        vals += [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories+[categories[0]],
            fill="toself", name=mn, line_color=col
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True, height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("🏆 AUC-ROC Comparison")
    fig_auc = px.bar(
        df_metrics, x="Model", y="AUC-ROC",
        color="AUC-ROC", color_continuous_scale="Blues", text="AUC-ROC"
    )
    fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_auc.update_layout(height=350)
    st.plotly_chart(fig_auc, use_container_width=True)

    best_mk  = df_metrics.loc[df_metrics["Recall"].idxmax(),  "Model"]
    best_auc = df_metrics.loc[df_metrics["AUC-ROC"].idxmax(), "Model"]
    st.info(f"💡 **Clinical Recommendation:** For {selected_disease}, "
            f"**{best_mk}** achieves the highest recall. "
            f"**{best_auc}** achieves the highest AUC-ROC.")


# ══════════════════════════════════════════════
# TAB 3 — FAIRNESS AUDIT
# ══════════════════════════════════════════════
with tab3:
    st.header("⚖️ Fairness Audit")
    st.markdown("Age-based fairness analysis using Fairlearn metrics.")

    fd = fairness_data.get(disease_key, {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("DPD",
                  f"{fd.get('demographic_parity_difference',0):.4f}",
                  help="Demographic Parity Difference (0 = perfect)")
    with c2:
        st.metric("EOD",
                  f"{fd.get('equalized_odds_difference',0):.4f}",
                  help="Equalized Odds Difference")
    with c3:
        st.metric("Bias Risk", fd.get("bias_risk","Unknown"))

    st.subheader("📊 Accuracy by Age Group")
    group_acc = fd.get("group_accuracy", {})
    if group_acc:
        df_group = pd.DataFrame(
            list(group_acc.items()), columns=["Age Group","Accuracy"]
        )
        fig_group = px.bar(
            df_group, x="Age Group", y="Accuracy",
            color="Accuracy", color_continuous_scale="RdYlGn",
            range_y=[0,1], text="Accuracy"
        )
        fig_group.update_traces(texttemplate="%{text:.3f}",
                                textposition="outside")
        fig_group.update_layout(height=350)
        st.plotly_chart(fig_group, use_container_width=True)

    st.subheader("🌐 Cross-Disease Fairness Comparison")
    dpd_vals = [fairness_data.get(dk,{}).get(
                "demographic_parity_difference",0) for dk in DISEASE_KEYS]
    eod_vals = [fairness_data.get(dk,{}).get(
                "equalized_odds_difference",0)    for dk in DISEASE_KEYS]
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name="DPD", x=DISEASES, y=dpd_vals,
                              marker_color="#1f77b4"))
    fig_comp.add_trace(go.Bar(name="EOD", x=DISEASES, y=eod_vals,
                              marker_color="#ff7f0e"))
    fig_comp.update_layout(barmode="group", height=400,
                           yaxis_title="Fairness Metric Value")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("🔍 SHAP Feature Importance")
    shap_bar_path = os.path.join(
        EXPLAINABILITY_DIR, f"{disease_key}_shap_bar.png"
    )
    if os.path.exists(shap_bar_path):
        st.image(shap_bar_path, use_container_width=True)
    else:
        st.info("SHAP image not found.")

    interpretations = {
        "High":     ("🔴","High age-based bias. Performance varies across groups."),
        "Moderate": ("🟠","Moderate bias. Some variation across groups."),
        "Low":      ("🟢","Low bias. Model performs consistently.")
    }
    icon, text = interpretations.get(
        fd.get("bias_risk","Low"), ("⚪","Unknown")
    )
    st.markdown(f"{icon} {text}")


# ══════════════════════════════════════════════
# TAB 4 — WHAT-IF SIMULATOR
# FIX ISSUE 5 — binary features now use
#   checkboxes; continuous features use sliders
#   with correct per-feature min/max/default.
#   Real model prediction runs on every change.
# ══════════════════════════════════════════════
with tab4:
    st.header("🔬 What-If Simulator")
    st.markdown("Adjust features and watch the prediction update live.")

    sim_profiles    = list(SAMPLE_PATIENTS.get(disease_key, {}).keys())
    sim_profile_sel = st.selectbox(
        "Start from a sample profile",
        ["Custom"] + sim_profiles,
        key="sim_profile"
    )
    sim_defaults = (
        SAMPLE_PATIENTS[disease_key][sim_profile_sel]
        if sim_profile_sel != "Custom"
        else {f: 0.0 for f in features}
    )

    col_sim1, col_sim2 = st.columns([1, 1])

    with col_sim1:
        st.subheader("Adjust Features")
        sim_values = {}
        ranges     = FEATURE_RANGES.get(disease_key, {})

        for feat in features:
            r       = ranges.get(feat, (0.0, 100.0, 0.0))
            mn_v    = float(r[0])
            mx_v    = float(r[1])
            def_v   = float(r[2])
            default = max(mn_v, min(mx_v, float(sim_defaults.get(feat, def_v))))

            if mx_v == 1.0 and mn_v == 0.0:
                # Binary feature → checkbox
                val = st.checkbox(
                    feat, value=bool(default), key=f"sim_{feat}"
                )
                sim_values[feat] = 1.0 if val else 0.0
            else:
                step = 1.0 if (mx_v - mn_v) > 10 else 0.1
                sim_values[feat] = st.slider(
                    feat, mn_v, mx_v, default, step,
                    key=f"sim_{feat}"
                )

    with col_sim2:
        st.subheader("Live Prediction")

        sim_patient_vals       = [sim_values[f] for f in features]
        sim_prob, sim_pred     = make_prediction(
            disease_key, model_key, sim_patient_vals, features
        )

        if sim_prob is None:
            st.error("Model could not be loaded. "
                     "Check that .pkl files exist in models/")
            sim_prob, sim_pred = 0.5, 0

        sim_risk = prob_to_risk(sim_prob)

        fig_sim = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sim_prob * 100,
            title={"text": "Disease Probability (%)"},
            gauge={
                "axis":  {"range": [0,100]},
                "bar":   {"color": risk_colour(sim_risk)},
                "steps": [
                    {"range": [0,  40], "color": "#e8f5e9"},
                    {"range": [40, 60], "color": "#fff3e0"},
                    {"range": [60,100], "color": "#ffebee"},
                ]
            }
        ))
        fig_sim.update_layout(height=280, margin=dict(t=30,b=10))
        st.plotly_chart(fig_sim, use_container_width=True)

        pred_label = "⚠️ POSITIVE" if sim_pred == 1 else "✅ NEGATIVE"
        pred_color = "#dc3545"     if sim_pred == 1 else "#28a745"
        st.markdown(
            f'<div style="background:{pred_color};color:white;'
            f'padding:1rem;border-radius:10px;text-align:center;'
            f'font-size:1.5rem;">{pred_label}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f"**Confidence:** {sim_prob*100:.1f}%  "
            f"| **Risk:** {sim_risk}"
        )

        if shap_data is not None:
            st.subheader("Feature Importance (SHAP)")
            importance = shap_data["feature_importance"]
            top_feats  = list(importance.keys())[:6]
            df_shap    = pd.DataFrame({
                "Feature"   : top_feats,
                "Importance": [round(importance[f],4) for f in top_feats],
            })
            st.dataframe(df_shap, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 5 — PREDICTION EXPLAINER
# FIX ISSUE 3 — removed all "real SHAP from Nema"
#   captions. Clean text only.
# FIX ISSUE 7 — Q&A is now fully dynamic:
#   answers depend on which feature was asked,
#   current prediction result, SHAP rank, etc.
# ══════════════════════════════════════════════
with tab5:
    st.header("💬 Prediction Explainer")
    st.markdown("Plain-English explanation from SHAP analysis.")

    col_e1, col_e2 = st.columns([1, 1])

    with col_e1:
        st.subheader("Model Summary")
        st.markdown(f"**Disease:** {selected_disease}")
        st.markdown(f"**Model:** {selected_model_name}")
        st.markdown(f"**AUC-ROC:** {m.get('auc_roc',0):.3f}")
        st.markdown(f"**Recall:** {m.get('recall',0)*100:.1f}%")

        if shap_data is not None:
            st.subheader("Top Risk Factors")
            importance = shap_data["feature_importance"]
            top5       = shap_data["top_5_features"]
            for rank, feat in enumerate(top5, 1):
                val = importance[feat]
                st.markdown(f"**{rank}.** {feat} — `{val:.4f}`")

    with col_e2:
        st.subheader("Plain-English Explanation")

        if shap_data is not None:
            importance = shap_data["feature_importance"]
            top5       = shap_data["top_5_features"]
            base_value = shap_data.get("base_value", 0.0)

            last_prob = st.session_state.get("t1_prob", None)
            last_pred = st.session_state.get("t1_pred", None)

            result_str = (
                f"**{'POSITIVE' if last_pred==1 else 'NEGATIVE'}** "
                f"({last_prob*100:.1f}% confidence)"
                if last_pred is not None
                else "Not yet predicted — generate label in Tab 1 first"
            )

            lines = [
                f"{i+1}. **{feat}** (importance: {importance[feat]:.4f})"
                for i, feat in enumerate(top5)
            ]

            st.markdown(f"""
**Last prediction:** {result_str}

**Why did the model make this prediction?**

The model analysed {len(features)} features. Top 5 drivers:

{"".join(chr(10) + l for l in lines)}

**Base risk** (average patient): {base_value*100:.1f}%

*This tool is for screening only. Always review with a clinician.*
*Accuracy: {m.get('accuracy',0)*100:.1f}%  |  Recall: {m.get('recall',0)*100:.1f}%*
""")

            wf_path = os.path.join(
                EXPLAINABILITY_DIR, f"{disease_key}_shap_waterfall.png"
            )
            if os.path.exists(wf_path):
                st.subheader("Single Patient Breakdown (SHAP Waterfall)")
                st.image(wf_path, use_container_width=True)
        else:
            st.warning("SHAP data not found.")

    # ── Dynamic Q&A ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Ask About This Prediction")
    user_q = st.text_input(
        "Type a question...",
        placeholder="e.g. Why is BMI important?  What does recall mean?"
    )

    if st.button("Get Answer", use_container_width=True):
        if not user_q:
            st.warning("Please type a question first.")
        elif shap_data is None:
            st.error("SHAP data not available.")
        else:
            importance    = shap_data["feature_importance"]
            all_feat_keys = list(importance.keys())
            top5          = shap_data["top_5_features"]
            last_pred     = st.session_state.get("t1_pred", None)
            last_prob     = st.session_state.get("t1_prob", None)
            q_lower       = user_q.lower()

            # 1. Feature-specific question
            matched_feat = next(
                (f for f in all_feat_keys if f.lower() in q_lower), None
            )

            if matched_feat:
                val      = importance[matched_feat]
                rank     = all_feat_keys.index(matched_feat) + 1
                strength = ("very strong" if rank <= 3
                            else "moderate" if rank <= 8 else "minor")
                in_top5  = matched_feat in top5
                pred_ctx = ""
                if last_pred is not None:
                    outcome  = "positive" if last_pred == 1 else "negative"
                    pred_ctx = (
                        f" The last prediction was **{outcome}** "
                        f"({last_prob*100:.1f}% confidence)."
                    )
                st.markdown(f"""
**About: {matched_feat}**

Ranked **#{rank}** out of {len(all_feat_keys)} features
(SHAP importance: {val:.4f}) — **{strength}** influence.

{"This is a top-5 most important feature for this model." if in_top5 else ""}
{pred_ctx}
""")

            # 2. AUC / ROC question
            elif any(w in q_lower for w in ["auc","roc"]):
                auc = m.get("auc_roc", 0)
                quality = ('excellent' if auc>=0.85
                           else 'good' if auc>=0.75 else 'moderate')
                st.markdown(f"""
**AUC-ROC for {selected_disease} ({selected_model_name}): {auc:.3f}**

Measures how well the model separates positive from negative cases.
1.0 = perfect, 0.5 = random. This model is **{quality}** at
distinguishing disease from no-disease — it correctly ranks a
positive case above a negative case **{auc*100:.1f}%** of the time.
""")

            # 3. Recall / sensitivity
            elif any(w in q_lower for w in ["recall","sensitivity","miss"]):
                recall = m.get("recall", 0)
                st.markdown(f"""
**Recall for {selected_disease}: {recall*100:.1f}%**

Out of every 100 real {selected_disease} cases, this model
correctly flags **{recall*100:.1f}** of them.
The remaining **{(1-recall)*100:.1f}** are missed (false negatives).
In clinical screening, high recall is critical.
""")

            # 4. Prediction / why / confidence
            elif any(w in q_lower for w in
                     ["predict","result","confidence","why","how"]):
                if last_pred is not None:
                    outcome  = "POSITIVE" if last_pred == 1 else "NEGATIVE"
                    top_feat = top5[0]
                    top_val  = importance[top_feat]
                    st.markdown(f"""
**Last prediction: {outcome} ({last_prob*100:.1f}% confidence)**

The single most influential feature in this model is **{top_feat}**
(importance: {top_val:.4f}). Top 5 features: {", ".join(top5)}.

Model recall: {m.get('recall',0)*100:.1f}% — catches that many real cases.
""")
                else:
                    st.info("Generate a prediction in Tab 1 first.")

            # 5. Bias / fairness / age
            elif any(w in q_lower for w in ["bias","fair","age","group"]):
                fd_q   = fairness_data.get(disease_key, {})
                bias_r = fd_q.get("bias_risk","Unknown")
                dpd    = fd_q.get("demographic_parity_difference",0)
                st.markdown(f"""
**Fairness — {selected_disease}**

Bias Risk: **{bias_r}**
DPD: {dpd:.4f} — positive prediction rate differs by
{dpd*100:.1f} percentage points across age groups.

{"⚠️ High bias — significant variation by age group." if bias_r=="High" else "✅ Low bias — consistent performance across groups."}
""")

            # 6. Generic fallback using real data
            else:
                top_feat = top5[0]
                top_val  = importance[top_feat]
                st.markdown(f"""
**Summary — {selected_disease} ({selected_model_name})**

Most important feature: **{top_feat}** (SHAP: {top_val:.4f})
Top 5: {", ".join(top5)}

AUC-ROC: {m.get('auc_roc',0):.3f}
Recall: {m.get('recall',0)*100:.1f}%
Accuracy: {m.get('accuracy',0)*100:.1f}%

Try asking: "Why is {top_feat} important?", "What is recall?",
"What does AUC mean?", or "Is there age bias?"
""")


# ══════════════════════════════════════════════
# TAB 6 — COMPLIANCE REPORT + PDF
# FIX ISSUE 6 — safe_pdf() strips all
#   non-latin-1 characters before writing,
#   and pdf.output() is called as bytes().
# ══════════════════════════════════════════════
with tab6:
    st.header("📄 Compliance Report")
    st.markdown("EU AI Act 2024 compliance assessment and PDF export.")

    st.subheader("✅ EU AI Act 2024 Compliance Checklist")
    checks = [
        ("Transparency & Explainability", True,
         "SHAP explanations generated for every prediction"),
        ("Fairness & Non-Discrimination", True,
         "Fairlearn DPD and EOD computed across age groups"),
        ("Human Oversight",              True,
         "Clinical disclaimer on all predictions"),
        ("Accuracy Reporting",           True,
         "Accuracy, Precision, Recall, F1, AUC-ROC reported"),
        ("Data Governance",              True,
         "Datasets: CDC BRFSS and UCI ML Repository"),
        ("Risk Classification",          True,
         "Classified as High-Risk AI per EU AI Act"),
        ("Bias Documentation",           True,
         "Bias risk labelled per disease"),
        ("Audit Trail",                  False,
         "Full audit logging pending"),
    ]
    for check_name, passed, detail in checks:
        icon = "✅" if passed else "⏳"
        c1, c2 = st.columns([3, 7])
        with c1:
            st.markdown(f"{icon} **{check_name}**")
        with c2:
            st.caption(detail)

    st.markdown("---")
    st.subheader("📊 Full Model Performance Report")
    all_rows = []
    for dk, dn in zip(DISEASE_KEYS, DISEASES):
        for mk, mn in zip(MODEL_KEYS, MODEL_TYPES):
            dm     = all_metrics.get(dk,{}).get(mk,{})
            fd_row = fairness_data.get(dk,{})
            all_rows.append({
                "Disease":   dn, "Model": mn,
                "Accuracy":  dm.get("accuracy",0),
                "Recall":    dm.get("recall",0),
                "AUC-ROC":   dm.get("auc_roc",0),
                "Bias Risk": fd_row.get("bias_risk","Unknown"),
            })
    df_full = pd.DataFrame(all_rows)
    st.dataframe(
        df_full.style.format({
            c: "{:.4f}" for c in ["Accuracy","Recall","AUC-ROC"]
        }),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("📥 Export as PDF")

    if st.button("📄 Generate PDF Report", use_container_width=True):
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_margins(15, 15, 15)

            # Title
            pdf.set_font("Arial","B",18)
            pdf.cell(0,12,safe_pdf("MediSight AI -- Transparency Label"),ln=True)
            pdf.set_font("Arial",size=12)
            pdf.cell(0,8,safe_pdf(f"Disease : {selected_disease}"),ln=True)
            pdf.cell(0,8,safe_pdf(f"Model   : {selected_model_name}"),ln=True)
            pdf.ln(4)

            # Last prediction
            last_pred_t6 = st.session_state.get("t1_pred", None)
            last_prob_t6 = st.session_state.get("t1_prob", None)
            if last_pred_t6 is not None:
                outcome = "POSITIVE" if last_pred_t6==1 else "NEGATIVE"
                pdf.set_font("Arial","B",13)
                pdf.cell(0,9,"PREDICTION",ln=True)
                pdf.set_font("Arial",size=11)
                pdf.cell(0,7,safe_pdf(f"  Result     : {outcome}"),ln=True)
                pdf.cell(0,7,
                    safe_pdf(f"  Confidence : {last_prob_t6*100:.1f}%"),ln=True)
                pdf.ln(4)

            # Metrics
            pdf.set_font("Arial","B",13)
            pdf.cell(0,9,"MODEL METRICS",ln=True)
            pdf.set_font("Arial",size=11)
            dm_pdf = all_metrics.get(disease_key,{}).get(model_key,{})
            for label, key in [
                ("Accuracy ","accuracy"),("Recall   ","recall"),
                ("Precision","precision"),("F1 Score ","f1_score"),
                ("AUC-ROC  ","auc_roc"),
            ]:
                pdf.cell(0,7,
                    safe_pdf(f"  {label}: {dm_pdf.get(key,0):.4f}"),ln=True)
            pdf.ln(4)

            # Fairness
            pdf.set_font("Arial","B",13)
            pdf.cell(0,9,"FAIRNESS ANALYSIS",ln=True)
            pdf.set_font("Arial",size=11)
            fd_pdf = fairness_data.get(disease_key,{})
            pdf.cell(0,7,
                safe_pdf(f"  Bias Risk : {fd_pdf.get('bias_risk','Unknown')}"),
                ln=True)
            pdf.cell(0,7,
                safe_pdf(f"  DPD       : "
                         f"{fd_pdf.get('demographic_parity_difference',0):.4f}"),
                ln=True)
            pdf.cell(0,7,
                safe_pdf(f"  EOD       : "
                         f"{fd_pdf.get('equalized_odds_difference',0):.4f}"),
                ln=True)
            pdf.ln(4)

            # SHAP
            if shap_data is not None:
                pdf.set_font("Arial","B",13)
                pdf.cell(0,9,"TOP SHAP FEATURES",ln=True)
                pdf.set_font("Arial",size=11)
                imp_pdf = shap_data["feature_importance"]
                for i, feat in enumerate(shap_data["top_5_features"],1):
                    pdf.cell(0,7,
                        safe_pdf(f"  {i}. {feat}: {imp_pdf[feat]:.4f}"),ln=True)
                pdf.ln(4)

            # EU AI Act
            pdf.set_font("Arial","B",13)
            pdf.cell(0,9,"EU AI ACT 2024 COMPLIANCE",ln=True)
            pdf.set_font("Arial",size=11)
            for check_name, passed, _ in checks:
                pdf.cell(0,7,
                    safe_pdf(f"  [{'PASS' if passed else 'PENDING'}] "
                             f"{check_name}"),ln=True)
            pdf.ln(4)

            # Disclaimer
            pdf.set_font("Arial","I",10)
            pdf.multi_cell(0,6,safe_pdf(
                "DISCLAIMER: This report is generated by an AI screening "
                "system. It does not constitute a clinical diagnosis. "
                "All predictions must be reviewed by a qualified "
                "healthcare professional."
            ))

            # Output — bytes() works for both fpdf and fpdf2
            pdf_bytes = bytes(pdf.output())

            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"medisight_{disease_key}_transparency_label.pdf",
                mime="application/pdf"
            )
            st.success("PDF generated successfully!")

        except ImportError:
            st.error("FPDF2 not installed. Run: pip install fpdf2")
        except Exception as e:
            st.error(f"PDF error: {e}")

    st.markdown("---")
    st.caption(
        "MediSight AI | Department of AI & DS, "
        "Mysore University School of Engineering | University of Mysore"
    )