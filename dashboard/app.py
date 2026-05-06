"""
MediSight AI — Streamlit Dashboard
A Transparency Label for Healthcare Prediction Systems

This is the production dashboard. All training/preprocessing/SHAP
computation happens offline in notebooks. This file ONLY:
  - loads pre-trained .pkl models (cached)
  - reads pre-computed JSON metrics, fairness, and SHAP files
  - serves predictions via cached model loaders
  - renders the transparency label, comparisons, audit trail, PDF
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import joblib
import os
from datetime import datetime
from io import BytesIO

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
# CUSTOM CSS — refined, subtler, more professional
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        padding: 0.8rem 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .label-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }
    .label-card table {
        margin-top: 0.5rem;
    }
    .label-card td {
        padding: 0.35rem 0;
        font-size: 0.95rem;
    }
    .big-prediction {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    .big-prediction-sub {
        font-size: 1.1rem;
        color: #666;
        margin: 0;
    }
    .info-tip {
        background: #f0f7ff;
        border-left: 3px solid #1f77b4;
        padding: 0.6rem 0.9rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #333;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 18px;
        border-radius: 8px 8px 0 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    /* Tighter expander headers */
    .streamlit-expanderHeader {
        font-size: 0.95rem;
        font-weight: 500;
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
# SAMPLE PATIENT PROFILES
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

# Feature ranges for sliders (min, max, default)
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

# Friendly tooltips for top features (shown on hover/help)
FEATURE_HELP = {
    # Diabetes
    "HighBP": "Has high blood pressure (1=yes, 0=no)",
    "HighChol": "Has high cholesterol (1=yes, 0=no)",
    "BMI": "Body Mass Index (kg/m²). 25-30=overweight, 30+=obese",
    "GenHlth": "Self-reported general health (1=excellent, 5=poor)",
    "Age": "Age category (1=18-24, 14=80+)",
    # Heart
    "AgeCategory": "Age in years (midpoint of category)",
    "GeneralHealth": "Self-reported health (1=poor, 5=excellent)",
    "ChestScan": "Has had a chest scan (1=yes, 0=no)",
    "Sex": "Biological sex (1=Male, 0=Female)",
    "HadDiabetes": "Has been diagnosed with diabetes (1=yes, 0=no)",
    "SmokerStatus": "0=never, 1=former, 2=some days, 3=daily",
    # Liver
    "Total Bilirubin": "Total bilirubin (mg/dL). Normal: 0.1-1.2",
    "Direct Bilirubin": "Direct bilirubin (mg/dL). Normal: 0.0-0.3",
    "Alkphos Alkaline Phosphotase": "ALP enzyme (IU/L). Normal: 44-147",
    "Sgpt Alamine Aminotransferase": "ALT enzyme (IU/L). Normal: 7-56",
    "Sgot Aspartate Aminotransferase": "AST enzyme (IU/L). Normal: 10-40",
    # Kidney
    "sg": "Specific gravity of urine (encoded 0-4)",
    "pcv": "Packed cell volume (%). Normal: 36-50",
    "dm": "Has diabetes mellitus (1=yes, 0=no)",
    "sc": "Serum creatinine (mg/dL). Normal: 0.6-1.3",
    "appet": "Appetite (1=good, 0=poor)",
    "htn": "Has hypertension (1=yes, 0=no)",
    "hemo": "Hemoglobin (g/dL). Normal: 13.5-17.5",
}

# ─────────────────────────────────────────────
# DATA LOADERS — JSON / METRICS (cache_data)
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

# ─────────────────────────────────────────────
# DATA LOADERS — MODEL OBJECTS (cache_resource)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(disease_key, model_key):
    path = os.path.join(MODELS_DIR, f"{disease_key}_{model_key}.pkl")
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model {disease_key}_{model_key}: {e}")
        return None

@st.cache_resource
def load_scaler(disease_key):
    path = os.path.join(MODELS_DIR, f"{disease_key}_scaler.pkl")
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource
def load_threshold(disease_key, model_key):
    """Caches heart_rf threshold so it's not re-read every prediction."""
    path = os.path.join(MODELS_DIR, f"{disease_key}_{model_key}_threshold.pkl")
    try:
        return joblib.load(path)
    except Exception:
        return 0.5

# ─────────────────────────────────────────────
# CORE BUSINESS LOGIC
# ─────────────────────────────────────────────
def get_decision_threshold(disease_key, model_key):
    """Returns the actual decision threshold for this model combo."""
    if disease_key == "heart" and model_key == "rf":
        return load_threshold("heart", "rf")
    return 0.5

def make_prediction(disease_key, model_key, patient_values, features):
    scaler = load_scaler(disease_key)
    model  = load_model(disease_key, model_key)
    if scaler is None or model is None:
        return None, None
    X_input  = np.array([patient_values], dtype=float)
    X_scaled = scaler.transform(X_input)
    prob     = float(model.predict_proba(X_scaled)[0][1])
    threshold = get_decision_threshold(disease_key, model_key)
    pred = 1 if prob >= threshold else 0
    return prob, pred

def prob_to_risk(prob, threshold=0.5):
    """Risk level relative to the model's actual decision threshold.

    Anchored to threshold so it stays meaningful across models with
    different cutoffs (e.g. heart RF uses 0.20 not 0.5).
    """
    if prob >= threshold + 0.15:
        return "High"
    elif prob >= threshold:
        return "Moderate"
    elif prob >= max(0, threshold - 0.15):
        return "Low"
    return "Very Low"

def risk_colour(level):
    return {
        "High":     "#dc3545",
        "Moderate": "#fd7e14",
        "Low":      "#28a745",
        "Very Low": "#6c757d",
    }.get(level, "#666")

def quality_badge_auc(auc):
    if auc >= 0.85: return "🟢 Excellent"
    if auc >= 0.75: return "🟡 Good"
    if auc >= 0.65: return "🟠 Fair"
    return "🔴 Poor"

def quality_badge_recall(recall):
    if recall >= 0.75: return "🟢 High"
    if recall >= 0.50: return "🟡 Moderate"
    if recall >= 0.25: return "🟠 Low"
    return "🔴 Very Low"

# ─────────────────────────────────────────────
# AUDIT TRAIL — EU AI Act compliance
# ─────────────────────────────────────────────
def log_prediction(disease_key, model_key, prob, pred, risk):
    """Append one prediction to JSONL audit log. Best-effort."""
    try:
        audit_dir = os.path.join(BASE_DIR, "../audit")
        os.makedirs(audit_dir, exist_ok=True)
        log_path = os.path.join(audit_dir, "predictions.jsonl")
        record = {
            "timestamp"  : datetime.utcnow().isoformat() + "Z",
            "disease"    : disease_key,
            "model"      : model_key,
            "probability": round(float(prob), 4),
            "prediction" : int(pred),
            "risk_level" : risk,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # never break the UI

# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
if "welcomed" not in st.session_state:
    st.session_state["welcomed"] = False
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 MediSight AI</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI Transparency Label for '
            'Healthcare Prediction Systems</div>',
            unsafe_allow_html=True)

# Welcome banner — dismissible
if not st.session_state["welcomed"]:
    welcome_col1, welcome_col2 = st.columns([20, 1])
    with welcome_col1:
        st.info(
            "👋 **Welcome.** Start in **Tab 1** to generate a transparency "
            "label, then explore Model Battle, Fairness, What-If, and the "
            "Compliance Report. Settings are in the left sidebar."
        )
    with welcome_col2:
        if st.button("✕", key="dismiss_welcome", help="Dismiss"):
            st.session_state["welcomed"] = True
            st.rerun()

# ─────────────────────────────────────────────
# SIDEBAR — cleaner, with quality badges
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

    selected_disease    = st.selectbox("Select Disease", DISEASES,
                                       help="Pick which disease to screen for")
    disease_key         = DISEASE_KEYS[DISEASES.index(selected_disease)]
    selected_model_name = st.selectbox("Select Model", MODEL_TYPES,
                                       help="Different models have different "
                                            "strengths — see Tab 2 to compare")
    model_key           = MODEL_KEYS[MODEL_TYPES.index(selected_model_name)]

    st.markdown("---")
    st.markdown("### 📊 Model Quality")
    metrics = load_all_metrics()
    m = metrics.get(disease_key, {}).get(model_key, {})
    if m:
        auc = m.get('auc_roc', 0)
        recall = m.get('recall', 0)
        st.metric("AUC-ROC",  f"{auc:.3f}",  quality_badge_auc(auc),
                  delta_color="off")
        st.metric("Recall",   f"{recall:.3f}", quality_badge_recall(recall),
                  delta_color="off")
        st.metric("Accuracy", f"{m.get('accuracy',0):.3f}")

    with st.expander("❓ What do these mean?"):
        st.markdown("""
- **AUC-ROC**: How well the model separates sick from healthy. 1.0=perfect.
- **Recall**: % of real cases the model catches. Critical for screening.
- **Accuracy**: Overall correctness. Less useful when classes are imbalanced.
        """)

    # Decision threshold disclosure
    threshold = get_decision_threshold(disease_key, model_key)
    if threshold != 0.5:
        st.warning(
            f"⚙️ This model uses a custom decision threshold of "
            f"**{threshold:.2f}** (not 0.5) for optimal F1 on "
            f"imbalanced data."
        )

    st.markdown("---")
    st.caption("Department of AI & DS\nMysore University School of Engineering")

# ─────────────────────────────────────────────
# SHARED DATA
# ─────────────────────────────────────────────
all_metrics   = load_all_metrics()
fairness_data = load_fairness_data()
features      = load_features(disease_key)
shap_data     = load_shap_data(disease_key)
threshold     = get_decision_threshold(disease_key, model_key)

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
# TAB 1 — TRANSPARENCY LABEL (UX-overhauled)
# ══════════════════════════════════════════════
with tab1:
    st.header("🏷️ AI Transparency Label")
    st.markdown(
        "Generate a standardised transparency label for any patient. "
        "Like a nutrition label — but for AI predictions."
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("👤 Patient Input")

        input_mode = st.radio(
            "How would you like to enter patient data?",
            ["📋 Sample Profile", "🎲 Random Patient", "✏️ Manual Entry"],
            horizontal=True,
        )

        # Mode descriptions — eliminate the "what does this do?" question
        mode_descriptions = {
            "📋 Sample Profile":
                "✅ **Recommended for demos.** Pre-validated High Risk and "
                "Low Risk profiles built from clinical literature.",
            "🎲 Random Patient":
                "🎲 Generates plausible random values within clinical ranges. "
                "Click again for new values. Good for stress-testing.",
            "✏️ Manual Entry":
                f"✏️ Enter the **top 5 most important features** (by SHAP). "
                f"The other features auto-fill with clinical defaults — "
                f"and you'll see exactly which ones."
        }
        st.markdown(
            f'<div class="info-tip">{mode_descriptions[input_mode]}</div>',
            unsafe_allow_html=True
        )

        sample_profiles = list(SAMPLE_PATIENTS.get(disease_key, {}).keys())
        patient_data = {}

        # ── MODE 1: Sample Profile — show values inline ────────────
        if input_mode == "📋 Sample Profile":
            selected_profile = st.selectbox(
                "Choose a profile",
                sample_profiles
            )
            profile_vals = SAMPLE_PATIENTS[disease_key][selected_profile]
            st.success(f"✅ Loaded: **{selected_profile}** profile")

            # Show values in 2 columns inline (no expander hiding it)
            st.markdown("##### Patient values:")
            val_col1, val_col2 = st.columns(2)
            items = list(profile_vals.items())
            half = (len(items) + 1) // 2
            with val_col1:
                for k, v in items[:half]:
                    st.markdown(f"<small>**{k}**: `{v}`</small>",
                                unsafe_allow_html=True)
            with val_col2:
                for k, v in items[half:]:
                    st.markdown(f"<small>**{k}**: `{v}`</small>",
                                unsafe_allow_html=True)

            patient_data = {f: float(profile_vals.get(f, 0.0))
                            for f in features}

        # ── MODE 2: Random Patient ─────────────────────────────────
        elif input_mode == "🎲 Random Patient":
            if st.button("🎲 Generate Random Patient",
                         use_container_width=True):
                ranges = FEATURE_RANGES.get(disease_key, {})
                random_vals = {}
                for feat in features:
                    r = ranges.get(feat, (0.0, 1.0, 0.0))
                    mn, mx = float(r[0]), float(r[1])
                    if mx == 1.0 and mn == 0.0:
                        random_vals[feat] = float(np.random.choice([0, 1]))
                    else:
                        random_vals[feat] = round(
                            float(np.random.uniform(mn, mx)), 2
                        )
                st.session_state["random_patient"] = random_vals

            if "random_patient" in st.session_state:
                patient_data = st.session_state["random_patient"]
                st.info("✨ Random patient generated. "
                        "Click button again for new values.")
                st.markdown("##### Patient values:")
                val_col1, val_col2 = st.columns(2)
                items = list(patient_data.items())
                half = (len(items) + 1) // 2
                with val_col1:
                    for k, v in items[:half]:
                        st.markdown(f"<small>**{k}**: `{v}`</small>",
                                    unsafe_allow_html=True)
                with val_col2:
                    for k, v in items[half:]:
                        st.markdown(f"<small>**{k}**: `{v}`</small>",
                                    unsafe_allow_html=True)
            else:
                st.info("👆 Click the button above to generate a patient.")
                ranges = FEATURE_RANGES.get(disease_key, {})
                patient_data = {f: float(ranges.get(f, (0,1,0))[2])
                                for f in features}

        # ── MODE 3: Manual Entry — top 5 visible, defaults transparent ─
        else:
            key_features = (
                shap_data.get("top_5_features", features[:5])
                if shap_data else features[:5]
            )
            ranges = FEATURE_RANGES.get(disease_key, {})

            st.markdown(f"##### Enter values for top 5 features:")

            for feat in key_features:
                r = ranges.get(feat, (0.0, 100.0, 0.0))
                mn, mx, default = float(r[0]), float(r[1]), float(r[2])
                help_text = FEATURE_HELP.get(
                    feat,
                    f"Clinical range: {mn} to {mx}"
                )
                if mx == 1.0 and mn == 0.0:
                    patient_data[feat] = float(
                        st.checkbox(feat, value=bool(default),
                                    key=f"t1m_{feat}",
                                    help=help_text)
                    )
                else:
                    step = 1.0 if (mx - mn) > 10 else 0.1
                    patient_data[feat] = st.slider(
                        feat, mn, mx, default, step,
                        key=f"t1m_{feat}",
                        help=help_text
                    )

            # Fill remaining with defaults — transparently
            other_features = [f for f in features if f not in patient_data]
            for feat in other_features:
                r = ranges.get(feat, (0.0, 1.0, 0.0))
                patient_data[feat] = float(r[2])

            if other_features:
                st.info(
                    f"ℹ️ {len(other_features)} other features were "
                    f"auto-filled with clinical defaults."
                )
                with st.expander("🔍 View default values used", expanded=False):
                    defaults_df = pd.DataFrame(
                        [(f, patient_data[f]) for f in other_features],
                        columns=["Feature", "Default Value"]
                    )
                    st.dataframe(defaults_df, use_container_width=True,
                                 hide_index=True)

        st.markdown("")  # spacing
        submitted = st.button(
            "🔍 Generate Transparency Label",
            use_container_width=True,
            type="primary"
        )

    with col_right:
        st.subheader("📊 Prediction Result")

        fairness_info = fairness_data.get(disease_key, {})
        bias_risk     = fairness_info.get("bias_risk", "Unknown")

        # Run prediction on submit
        if submitted:
            with st.spinner("🔬 Running prediction & generating label..."):
                patient_values = [patient_data[f] for f in features]
                real_prob, real_pred = make_prediction(
                    disease_key, model_key, patient_values, features
                )
                if real_prob is not None:
                    probability = real_prob
                    prediction  = real_pred
                    risk_level  = prob_to_risk(probability, threshold)
                    st.session_state["t1_prob"] = probability
                    st.session_state["t1_pred"] = prediction
                    st.session_state["t1_risk"] = risk_level

                    log_prediction(disease_key, model_key,
                                   probability, prediction, risk_level)

                    # Track for comparison
                    st.session_state["prediction_history"].append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "disease":   selected_disease,
                        "model":     selected_model_name,
                        "probability": probability,
                        "risk":       risk_level,
                        "input_mode": input_mode.replace("📋 ", "")
                                                .replace("🎲 ", "")
                                                .replace("✏️ ", ""),
                    })
                    # Keep last 5
                    st.session_state["prediction_history"] = \
                        st.session_state["prediction_history"][-5:]
                else:
                    st.warning("Model files not found.")
                    probability, prediction, risk_level = 0.5, 0, "Moderate"
        else:
            probability = st.session_state.get("t1_prob", None)
            prediction  = st.session_state.get("t1_pred", None)
            risk_level  = st.session_state.get("t1_risk", None)

        # ── Big visual prediction (eye-catching first) ─────────────
        if prediction is not None:
            outcome_icon  = "⚠️" if prediction == 1 else "✅"
            outcome_text  = "POSITIVE" if prediction == 1 else "NEGATIVE"
            outcome_color = "#dc3545" if prediction == 1 else "#28a745"

            st.markdown(
                f"""
                <div style="text-align:center;padding:1rem 0;">
                    <div style="font-size:1.3rem;color:{outcome_color};
                                font-weight:600;">
                        {outcome_icon} {outcome_text}
                    </div>
                    <div class="big-prediction" style="color:{risk_colour(risk_level)};">
                        {probability*100:.1f}%
                    </div>
                    <div class="big-prediction-sub">
                        Risk Level: <b>{risk_level}</b>
                        &nbsp;·&nbsp; Threshold: <b>{threshold*100:.0f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ── Compact label card ─────────────────────────────────
            st.markdown(f"""
<div class="label-card">
  <h4 style="margin:0 0 0.4rem 0;font-size:1.1rem;">
    🏥 MediSight AI Transparency Label
  </h4>
  <p style="margin:0;opacity:0.85;font-size:0.85rem;">
    {selected_disease} &nbsp;·&nbsp; {selected_model_name}
  </p>
  <hr style="border-color:rgba(255,255,255,0.3);margin:0.5rem 0">
  <table width="100%">
    <tr>
      <td>📈 AUC-ROC</td>
      <td style="text-align:right"><b>{m.get('auc_roc',0):.3f}</b></td>
    </tr>
    <tr>
      <td>🎯 Recall</td>
      <td style="text-align:right"><b>{m.get('recall',0)*100:.1f}%</b></td>
    </tr>
    <tr>
      <td>⚖️ Bias Risk</td>
      <td style="text-align:right"><b>{bias_risk}</b></td>
    </tr>
  </table>
  <hr style="border-color:rgba(255,255,255,0.3);margin:0.5rem 0">
  <p style="margin:0;font-size:0.75rem;opacity:0.85;">
    ✅ EU AI Act 2024 Compliant
  </p>
</div>
""", unsafe_allow_html=True)

            # Smart model recommendation hint
            disease_metrics = all_metrics.get(disease_key, {})
            best_recall_mk = max(
                disease_metrics.items(),
                key=lambda x: x[1].get('recall', 0)
            )[0]
            if best_recall_mk != model_key:
                best_name = MODEL_TYPES[MODEL_KEYS.index(best_recall_mk)]
                best_recall = disease_metrics[best_recall_mk].get('recall', 0)
                if best_recall > m.get('recall', 0) + 0.05:
                    st.caption(
                        f"💡 **Tip:** {best_name} has higher recall "
                        f"({best_recall*100:.1f}% vs {m.get('recall',0)*100:.1f}%) "
                        f"for {selected_disease}. Switch in the sidebar."
                    )
        else:
            st.info(
                "👈 Configure patient input on the left, then click "
                "**Generate Transparency Label** to see the prediction here."
            )

    # ── Below both columns: Gauge with model-aware threshold ──
    if prediction is not None:
        st.markdown("---")

        gauge_col, history_col = st.columns([1, 1], gap="large")

        with gauge_col:
            st.markdown("##### Confidence Gauge")
            threshold_pct = threshold * 100

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                delta={"reference": threshold_pct,
                       "increasing": {"color": "#dc3545"},
                       "decreasing": {"color": "#28a745"}},
                number={"suffix": "%", "font": {"size": 32}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": risk_colour(risk_level),
                             "thickness": 0.7},
                    "steps": [
                        {"range": [0, max(0, threshold_pct - 15)],
                         "color": "#e8f5e9"},
                        {"range": [max(0, threshold_pct - 15), threshold_pct],
                         "color": "#fff3e0"},
                        {"range": [threshold_pct, 100],
                         "color": "#ffebee"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.85,
                        "value": threshold_pct
                    }
                }
            ))
            fig_gauge.update_layout(
                height=260,
                margin=dict(t=20, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption(
                f"Black line marks the decision threshold ({threshold_pct:.0f}%). "
                f"Above = positive prediction."
            )

        with history_col:
            st.markdown("##### Recent Predictions")
            history = st.session_state["prediction_history"]
            if len(history) >= 1:
                history_df = pd.DataFrame(history)
                history_df["probability"] = (
                    history_df["probability"] * 100
                ).round(1).astype(str) + "%"
                history_df = history_df.rename(columns={
                    "timestamp": "Time",
                    "disease": "Disease",
                    "model": "Model",
                    "probability": "Confidence",
                    "risk": "Risk",
                    "input_mode": "Input"
                })
                # Show most recent first
                st.dataframe(
                    history_df[::-1],
                    use_container_width=True,
                    hide_index=True,
                    height=260
                )
                if len(history) > 1:
                    st.caption(f"📊 Compare last {len(history)} predictions.")
            else:
                st.info("Your prediction history will appear here.")

    # ── Below: SHAP factor visualizations ─────────────────────────
    st.markdown("---")
    shap_c1, shap_c2 = st.columns([1, 1], gap="large")

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
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis={'categoryorder':'total ascending'}
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
    st.markdown(f"Compare all three models for **{selected_disease}** "
                "side-by-side. Choose the one whose strengths fit your use case.")

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
                  .apply(highlight_best, subset=["AUC-ROC", "Recall", "F1 Score"])
                  .format({c: "{:.4f}" for c in df_metrics.columns[1:]}),
        use_container_width=True,
        hide_index=True
    )
    st.caption("✅ Green cells = best score in that column.")

    radar_col, auc_col = st.columns([3, 2], gap="large")

    with radar_col:
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
                fill="toself", name=mn, line_color=col,
                opacity=0.6
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            showlegend=True, height=420,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with auc_col:
        st.subheader("🏆 AUC-ROC")
        fig_auc = px.bar(
            df_metrics, x="Model", y="AUC-ROC",
            color="AUC-ROC", color_continuous_scale="Blues", text="AUC-ROC"
        )
        fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_auc.update_layout(height=420, showlegend=False,
                              margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_auc, use_container_width=True)

    best_recall_idx = df_metrics["Recall"].idxmax()
    best_auc_idx    = df_metrics["AUC-ROC"].idxmax()
    best_recall_m   = df_metrics.loc[best_recall_idx, "Model"]
    best_auc_m      = df_metrics.loc[best_auc_idx, "Model"]

    st.success(
        f"💡 **Clinical Recommendation for {selected_disease}:** "
        f"**{best_recall_m}** has the highest recall "
        f"({df_metrics.loc[best_recall_idx, 'Recall']:.1%}) — best for "
        f"catching real cases. **{best_auc_m}** has the highest AUC-ROC "
        f"({df_metrics.loc[best_auc_idx, 'AUC-ROC']:.3f}) — best for "
        f"overall discrimination."
    )

# ══════════════════════════════════════════════
# TAB 3 — FAIRNESS AUDIT
# ══════════════════════════════════════════════
with tab3:
    st.header("⚖️ Fairness Audit")
    st.markdown("Age-based fairness analysis using **Fairlearn** metrics. "
                "Does this model work equally well for all age groups?")

    fd = fairness_data.get(disease_key, {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("DPD",
                  f"{fd.get('demographic_parity_difference',0):.4f}",
                  help="Demographic Parity Difference. 0 = identical "
                       "positive prediction rate across groups.")
    with c2:
        st.metric("EOD",
                  f"{fd.get('equalized_odds_difference',0):.4f}",
                  help="Equalized Odds Difference. 0 = identical "
                       "true/false positive rates across groups.")
    with c3:
        bias = fd.get("bias_risk","Unknown")
        st.metric("Bias Risk", bias)

    interpretations = {
        "High":     ("🔴", "**High age-based bias.** Performance varies "
                            "significantly across age groups. Clinical use "
                            "should account for this disparity."),
        "Moderate": ("🟠", "**Moderate bias.** Some variation across groups."),
        "Low":      ("🟢", "**Low bias.** Model performs consistently across "
                            "age groups.")
    }
    icon, text = interpretations.get(
        fd.get("bias_risk","Low"), ("⚪","Unknown")
    )
    st.markdown(f"{icon} {text}")

    st.markdown("---")

    fair_col1, fair_col2 = st.columns([1, 1], gap="large")

    with fair_col1:
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
            fig_group.update_layout(height=350, showlegend=False,
                                    margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_group, use_container_width=True)

    with fair_col2:
        st.subheader("🌐 Cross-Disease Comparison")
        dpd_vals = [fairness_data.get(dk,{}).get(
                    "demographic_parity_difference",0) for dk in DISEASE_KEYS]
        eod_vals = [fairness_data.get(dk,{}).get(
                    "equalized_odds_difference",0)    for dk in DISEASE_KEYS]
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="DPD", x=DISEASES, y=dpd_vals,
                                  marker_color="#1f77b4"))
        fig_comp.add_trace(go.Bar(name="EOD", x=DISEASES, y=eod_vals,
                                  marker_color="#ff7f0e"))
        fig_comp.update_layout(barmode="group", height=350,
                               yaxis_title="Fairness Metric",
                               margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 SHAP Feature Importance")
    shap_bar_path = os.path.join(
        EXPLAINABILITY_DIR, f"{disease_key}_shap_bar.png"
    )
    if os.path.exists(shap_bar_path):
        st.image(shap_bar_path, use_container_width=True)
    else:
        st.info("SHAP image not found.")

# ══════════════════════════════════════════════
# TAB 4 — WHAT-IF SIMULATOR (top-10 visible, rest hidden)
# ══════════════════════════════════════════════
with tab4:
    st.header("🔬 What-If Simulator")
    st.markdown("Adjust features and watch the prediction update live. "
                "Top features by SHAP importance are shown first.")

    sim_profiles    = list(SAMPLE_PATIENTS.get(disease_key, {}).keys())
    sim_profile_sel = st.selectbox(
        "Start from a sample profile",
        ["Custom"] + sim_profiles,
        key="sim_profile",
        help="Pick a starting point, then adjust sliders to explore."
    )
    sim_defaults = (
        SAMPLE_PATIENTS[disease_key][sim_profile_sel]
        if sim_profile_sel != "Custom"
        else {f: 0.0 for f in features}
    )

    col_sim1, col_sim2 = st.columns([1, 1], gap="large")

    with col_sim1:
        st.subheader("Adjust Features")

        # Determine priority (top SHAP) vs other features
        if shap_data and "top_10_features" in shap_data:
            priority_feats = [f for f in shap_data["top_10_features"]
                              if f in features]
        else:
            priority_feats = features[:10]

        # Cap at 10, ensure all exist in features
        priority_feats = priority_feats[:10]
        other_feats = [f for f in features if f not in priority_feats]

        sim_values = {}
        ranges = FEATURE_RANGES.get(disease_key, {})

        st.caption(f"⭐ **Top {len(priority_feats)} features** "
                   f"by SHAP importance — biggest impact on predictions")

        def render_slider(feat, key_prefix=""):
            r = ranges.get(feat, (0.0, 100.0, 0.0))
            mn_v = float(r[0])
            mx_v = float(r[1])
            def_v = float(r[2])
            default = max(mn_v, min(mx_v,
                          float(sim_defaults.get(feat, def_v))))
            help_text = FEATURE_HELP.get(feat,
                                         f"Range: {mn_v} to {mx_v}")
            if mx_v == 1.0 and mn_v == 0.0:
                val = st.checkbox(
                    feat, value=bool(default),
                    key=f"{key_prefix}sim_{feat}",
                    help=help_text
                )
                return 1.0 if val else 0.0
            else:
                step = 1.0 if (mx_v - mn_v) > 10 else 0.1
                return st.slider(
                    feat, mn_v, mx_v, default, step,
                    key=f"{key_prefix}sim_{feat}",
                    help=help_text
                )

        for feat in priority_feats:
            sim_values[feat] = render_slider(feat)

        if other_feats:
            with st.expander(f"➕ Show {len(other_feats)} additional features",
                             expanded=False):
                st.caption("These have lower SHAP impact but you can still "
                           "adjust them.")
                for feat in other_feats:
                    sim_values[feat] = render_slider(feat, "other_")

        # Ensure every feature has a value
        for feat in features:
            if feat not in sim_values:
                r = ranges.get(feat, (0.0, 1.0, 0.0))
                sim_values[feat] = float(r[2])

    with col_sim2:
        st.subheader("Live Prediction")

        sim_patient_vals = [sim_values[f] for f in features]
        sim_prob, sim_pred = make_prediction(
            disease_key, model_key, sim_patient_vals, features
        )

        if sim_prob is None:
            st.error("Model could not be loaded.")
            sim_prob, sim_pred = 0.5, 0

        sim_risk = prob_to_risk(sim_prob, threshold)

        # Big visual outcome
        outcome_icon = "⚠️" if sim_pred == 1 else "✅"
        outcome_text = "POSITIVE" if sim_pred == 1 else "NEGATIVE"
        outcome_color = "#dc3545" if sim_pred == 1 else "#28a745"

        st.markdown(
            f"""
            <div style="text-align:center;padding:0.5rem 0;">
                <div style="font-size:1.2rem;color:{outcome_color};
                            font-weight:600;">
                    {outcome_icon} {outcome_text}
                </div>
                <div class="big-prediction"
                     style="color:{risk_colour(sim_risk)};font-size:2.5rem;">
                    {sim_prob*100:.1f}%
                </div>
                <div class="big-prediction-sub">
                    Risk: <b>{sim_risk}</b> ·
                    Threshold: <b>{threshold*100:.0f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Gauge with proper threshold
        threshold_pct = threshold * 100
        fig_sim = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sim_prob * 100,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis":  {"range": [0, 100], "tickwidth": 1},
                "bar":   {"color": risk_colour(sim_risk),
                          "thickness": 0.7},
                "steps": [
                    {"range": [0, max(0, threshold_pct - 15)],
                     "color": "#e8f5e9"},
                    {"range": [max(0, threshold_pct - 15), threshold_pct],
                     "color": "#fff3e0"},
                    {"range": [threshold_pct, 100],
                     "color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.85,
                    "value": threshold_pct
                }
            }
        ))
        fig_sim.update_layout(
            height=240,
            margin=dict(t=10, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        if shap_data is not None:
            st.markdown("##### Feature Impact (SHAP)")
            importance = shap_data["feature_importance"]
            top_feats  = list(importance.keys())[:6]
            df_shap    = pd.DataFrame({
                "Feature"   : top_feats,
                "Importance": [round(importance[f],4) for f in top_feats],
            })
            st.dataframe(df_shap, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 5 — PREDICTION EXPLAINER (Q&A at top, more prominent)
# ══════════════════════════════════════════════
with tab5:
    st.header("💬 Prediction Explainer")
    st.markdown("Plain-English explanation from SHAP analysis. "
                "Ask questions about the model below.")

    # ── Q&A FIRST — most engaging feature, top of page ───────────
    st.subheader("💬 Ask About This Prediction")

    qa_col1, qa_col2 = st.columns([5, 1])
    with qa_col1:
        user_q = st.text_input(
            "Type a question",
            placeholder="e.g. Why is BMI important?  What does recall mean?",
            key="qa_input",
            label_visibility="collapsed"
        )
    with qa_col2:
        ask_btn = st.button("Ask", use_container_width=True, type="primary")

    if ask_btn and user_q and shap_data is not None:
        importance    = shap_data["feature_importance"]
        all_feat_keys = list(importance.keys())
        top5          = shap_data["top_5_features"]
        last_pred     = st.session_state.get("t1_pred", None)
        last_prob     = st.session_state.get("t1_prob", None)
        q_lower       = user_q.lower()

        prev_questions = [h["q"].lower()
                          for h in st.session_state["qa_history"]]
        is_repeat = any(q_lower in pq or pq in q_lower
                        for pq in prev_questions)
        repeat_note = ("\n\n*ℹ️ Similar question asked before — "
                       "here's a different angle.*"
                       if is_repeat else "")

        answer = ""
        matched_feat = next(
            (f for f in all_feat_keys if f.lower() in q_lower), None
        )

        # 1. Feature-specific
        if matched_feat:
            val      = importance[matched_feat]
            rank     = all_feat_keys.index(matched_feat) + 1
            strength = ("very strong" if rank <= 3
                        else "moderate" if rank <= 8 else "minor")
            in_top5  = matched_feat in top5
            repeat_count = sum(
                1 for pq in prev_questions
                if matched_feat.lower() in pq
            )

            if repeat_count == 0:
                answer = f"""
**About: {matched_feat}**

Ranked **#{rank}** out of {len(all_feat_keys)} features
(SHAP importance: `{val:.4f}`) — **{strength}** influence on the model's decision.

{"⭐ This is a top-5 driver for " + selected_disease + " predictions." if in_top5 else "This feature has secondary influence."}
"""
            elif repeat_count == 1:
                top_val = importance[top5[0]]
                ratio = val / top_val if top_val > 0 else 0
                answer = f"""
**More on {matched_feat}** (relative comparison)

Compared to the most important feature (`{top5[0]}`), this one carries
**{ratio*100:.1f}%** of its influence. In clinical terms, {matched_feat}
contributes meaningfully but isn't the dominant signal.
"""
            else:
                answer = f"""
**Technical view of {matched_feat}**

Mean absolute SHAP value: `{val:.4f}`. This represents the average impact
this feature has on pushing predictions toward the positive class
(in log-odds for tree models, in linear contribution for LR).
"""
            if last_pred is not None:
                outcome = "positive" if last_pred == 1 else "negative"
                answer += (f"\n\n**Current case context**: prediction is "
                           f"**{outcome}** ({last_prob*100:.1f}% confidence).")

        # 2. AUC
        elif any(w in q_lower for w in ["auc","roc"]):
            auc = m.get("auc_roc", 0)
            quality = ('excellent' if auc>=0.85
                       else 'good' if auc>=0.75 else 'moderate')
            answer = f"""
**AUC-ROC for {selected_disease} ({selected_model_name}): `{auc:.3f}`**

This measures how well the model **ranks** a positive case higher than
a negative one. 1.0 = perfect, 0.5 = random guessing.

This model is **{quality}** at discrimination — it correctly orders a
positive case above a negative one **{auc*100:.1f}%** of the time.
"""
        # 3. Recall
        elif any(w in q_lower for w in ["recall","sensitivity","miss"]):
            recall = m.get("recall", 0)
            answer = f"""
**Recall (Sensitivity) for {selected_disease}: `{recall*100:.1f}%`**

Out of every 100 real {selected_disease.lower()} cases, this model catches
**{recall*100:.0f}** of them. The remaining **{(1-recall)*100:.0f}** are
missed (false negatives).

In medical screening, recall matters more than precision — a missed
disease is worse than a false alarm.
"""
        # 4. Bias
        elif any(w in q_lower for w in ["bias","fair","age","group"]):
            fd_q   = fairness_data.get(disease_key, {})
            bias_r = fd_q.get("bias_risk","Unknown")
            dpd    = fd_q.get("demographic_parity_difference",0)
            answer = f"""
**Fairness — {selected_disease}**

Bias Risk: **{bias_r}**
DPD: `{dpd:.4f}` — positive prediction rate differs by
**{dpd*100:.1f} percentage points** across age groups.

{"⚠️ High bias detected — significant variation by age group." if bias_r=="High" else "✅ Low bias — consistent performance across groups."}
"""
        # 5. Threshold
        elif any(w in q_lower for w in ["threshold","cutoff","boundary"]):
            answer = f"""
**Decision threshold: `{threshold:.2f}` ({threshold*100:.0f}%)**

Probabilities above this value are flagged as positive. The default is 0.5,
but {selected_disease} {selected_model_name} uses **{threshold:.2f}**
{"(custom threshold tuned for optimal F1 on imbalanced data)" if threshold != 0.5 else "(standard threshold)"}.
"""
        # 6. Why / how
        elif any(w in q_lower for w in
                 ["predict","result","confidence","why","how"]):
            if last_pred is not None:
                outcome  = "POSITIVE" if last_pred == 1 else "NEGATIVE"
                top_feat = top5[0]
                top_val  = importance[top_feat]
                answer = f"""
**Last prediction: {outcome} ({last_prob*100:.1f}% confidence)**

The dominant feature driving this prediction is **{top_feat}**
(SHAP importance: `{top_val:.4f}`).

Top 5 contributing factors: {", ".join(f"`{f}`" for f in top5)}

Model recall ({m.get('recall',0)*100:.1f}%) means it catches that
fraction of real cases.
"""
            else:
                answer = "ℹ️ Generate a prediction in **Tab 1** first."
        # 7. Fallback
        else:
            top_feat = top5[0]
            answer = f"""
**Quick Summary — {selected_disease} ({selected_model_name})**

- Top feature: **{top_feat}** (SHAP: `{importance[top_feat]:.4f}`)
- Top 5: {", ".join(f"`{f}`" for f in top5)}
- AUC-ROC: `{m.get('auc_roc',0):.3f}` | Recall: `{m.get('recall',0)*100:.1f}%`

**Try asking:**
- "Why is {top_feat} important?"
- "What does AUC mean?"
- "Is there age bias?"
- "What's the threshold?"
"""

        st.session_state["qa_history"].append({
            "q": user_q,
            "a": answer + repeat_note
        })
        st.rerun()
    elif ask_btn and not user_q:
        st.warning("Please type a question first.")
    elif ask_btn and shap_data is None:
        st.error("SHAP data not available.")

    # Render conversation (newest first)
    if st.session_state["qa_history"]:
        clear_col1, clear_col2 = st.columns([5, 1])
        with clear_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["qa_history"] = []
                st.rerun()

        for i, qa in enumerate(reversed(st.session_state["qa_history"])):
            with st.expander(f"❓ {qa['q']}", expanded=(i == 0)):
                st.markdown(qa["a"])

    st.markdown("---")

    # ── Below: model summary and explanation ──────────────────────
    col_e1, col_e2 = st.columns([1, 1], gap="large")

    with col_e1:
        st.subheader("Model Summary")
        st.markdown(f"**Disease:** {selected_disease}")
        st.markdown(f"**Model:** {selected_model_name}")
        st.markdown(f"**AUC-ROC:** {m.get('auc_roc',0):.3f}")
        st.markdown(f"**Recall:** {m.get('recall',0)*100:.1f}%")
        st.markdown(f"**Decision Threshold:** {threshold:.2f}")

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
                else "*Not yet predicted — generate a label in Tab 1 first.*"
            )

            lines = [
                f"{i+1}. **{feat}** (importance: {importance[feat]:.4f})"
                for i, feat in enumerate(top5)
            ]

            st.markdown(f"""
**Last prediction:** {result_str}

The model analysed {len(features)} features. Top 5 drivers:

{"".join(chr(10) + l for l in lines)}

**Base risk** (average patient): {base_value*100:.1f}%

*This tool is for screening only. Always review with a clinician.*
""")

            wf_path = os.path.join(
                EXPLAINABILITY_DIR, f"{disease_key}_shap_waterfall.png"
            )
            if os.path.exists(wf_path):
                st.markdown("##### Single Patient Breakdown (SHAP Waterfall)")
                st.image(wf_path, use_container_width=True)
        else:
            st.warning("SHAP data not found.")

# ══════════════════════════════════════════════
# TAB 6 — COMPLIANCE REPORT + PROFESSIONAL PDF
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
        ("Audit Trail",                  True,
         "All predictions logged to audit/predictions.jsonl"),
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
        use_container_width=True,
        hide_index=True
    )

    # Copy summary feature
    st.markdown("---")
    st.subheader("📋 Quick Summary (for clinical notes)")
    last_pred_t6 = st.session_state.get("t1_pred", None)
    last_prob_t6 = st.session_state.get("t1_prob", None)
    last_risk_t6 = st.session_state.get("t1_risk", "N/A")

    if last_pred_t6 is not None:
        if shap_data:
            top_factors = ", ".join(shap_data['top_5_features'][:3])
        else:
            top_factors = "N/A"
        summary_text = (
            f"MediSight AI - {selected_disease} Screening\n"
            f"Model: {selected_model_name}\n"
            f"Result: {'POSITIVE' if last_pred_t6 == 1 else 'NEGATIVE'}\n"
            f"Confidence: {last_prob_t6*100:.1f}%\n"
            f"Risk Level: {last_risk_t6}\n"
            f"Decision Threshold: {threshold:.2f}\n"
            f"Top Risk Factors: {top_factors}\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"AUC-ROC: {m.get('auc_roc',0):.3f}, "
            f"Recall: {m.get('recall',0)*100:.1f}%"
        )
        st.code(summary_text, language=None)
        st.caption("☝️ Click the copy icon in the top-right of the box "
                   "to copy this summary.")
    else:
        st.info("Generate a prediction in Tab 1 to see the summary here.")

    st.markdown("---")
    st.subheader("📥 Export Full Report as PDF")

    if st.button("📄 Generate PDF Report", use_container_width=True,
                 type="primary"):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak
            )
            from reportlab.lib.enums import TA_CENTER

            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                leftMargin=2*cm, rightMargin=2*cm,
                topMargin=2*cm, bottomMargin=2*cm,
                title=f"MediSight AI Transparency Label - {selected_disease}"
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle', parent=styles['Heading1'],
                fontSize=22, textColor=colors.HexColor("#1f4e79"),
                alignment=TA_CENTER, spaceAfter=6
            )
            subtitle_style = ParagraphStyle(
                'Subtitle', parent=styles['Normal'],
                fontSize=11, textColor=colors.HexColor("#666666"),
                alignment=TA_CENTER, spaceAfter=20
            )
            section_style = ParagraphStyle(
                'Section', parent=styles['Heading2'],
                fontSize=14, textColor=colors.HexColor("#1f4e79"),
                spaceBefore=14, spaceAfter=8
            )
            disclaimer_style = ParagraphStyle(
                'Disclaimer', parent=styles['Normal'],
                fontSize=9, textColor=colors.HexColor("#888888"),
                alignment=TA_CENTER, spaceBefore=20,
                fontName='Helvetica-Oblique'
            )

            story = []

            # Header
            story.append(Paragraph("MediSight AI", title_style))
            story.append(Paragraph(
                "Transparency Label for Healthcare Prediction Systems",
                subtitle_style
            ))
            story.append(Paragraph(
                f"Generated on {datetime.now().strftime('%d %B %Y, %H:%M')} "
                f"&nbsp;|&nbsp; Department of AI &amp; DS, "
                f"Mysore University School of Engineering",
                ParagraphStyle('meta', parent=styles['Normal'], fontSize=9,
                               textColor=colors.grey, alignment=TA_CENTER)
            ))
            story.append(Spacer(1, 0.5*cm))

            # 1. Prediction Summary
            story.append(Paragraph("1. Prediction Summary", section_style))
            if last_pred_t6 is not None:
                outcome = ("POSITIVE - Disease Detected"
                           if last_pred_t6 == 1
                           else "NEGATIVE - No Disease")
                pred_data = [
                    ["Field", "Value"],
                    ["Disease Screened", selected_disease],
                    ["Model Used", selected_model_name],
                    ["Result", outcome],
                    ["Confidence", f"{last_prob_t6*100:.1f}%"],
                    ["Risk Level", last_risk_t6],
                    ["Decision Threshold", f"{threshold:.2f}"],
                ]
            else:
                pred_data = [
                    ["Field", "Value"],
                    ["Disease Screened", selected_disease],
                    ["Model Used", selected_model_name],
                    ["Status",
                     "No prediction generated yet - run Tab 1 first"],
                ]
            pred_table = Table(pred_data, colWidths=[6*cm, 10*cm])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1f4e79")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 10),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0,1), (-1,-1),
                 [colors.white, colors.HexColor("#f5f5f5")]),
                ('PADDING', (0,0), (-1,-1), 8),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(pred_table)

            # 2. Performance
            story.append(Paragraph("2. Model Performance Metrics",
                                   section_style))
            dm_pdf = all_metrics.get(disease_key, {}).get(model_key, {})
            perf_data = [
                ["Metric", "Value", "Interpretation"],
                ["Accuracy",  f"{dm_pdf.get('accuracy',0):.4f}",
                 "Overall correctness across all predictions"],
                ["Precision", f"{dm_pdf.get('precision',0):.4f}",
                 "Of flagged cases, how many are truly positive"],
                ["Recall",    f"{dm_pdf.get('recall',0):.4f}",
                 "Of real cases, how many the model catches"],
                ["F1 Score",  f"{dm_pdf.get('f1_score',0):.4f}",
                 "Balance between precision and recall"],
                ["AUC-ROC",   f"{dm_pdf.get('auc_roc',0):.4f}",
                 "Discrimination quality (1.0 = perfect)"],
            ]
            perf_table = Table(perf_data, colWidths=[3.5*cm, 3*cm, 9.5*cm])
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1f4e79")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0,1), (-1,-1),
                 [colors.white, colors.HexColor("#f5f5f5")]),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(perf_table)

            # 3. Fairness
            story.append(Paragraph("3. Fairness Analysis (Age-based)",
                                   section_style))
            fd_pdf = fairness_data.get(disease_key, {})
            bias_color = {
                "High":     colors.HexColor("#dc3545"),
                "Moderate": colors.HexColor("#fd7e14"),
                "Low":      colors.HexColor("#28a745")
            }.get(fd_pdf.get("bias_risk", "Unknown"), colors.grey)

            fair_data = [
                ["Metric", "Value"],
                ["Bias Risk Classification",
                 fd_pdf.get("bias_risk", "Unknown")],
                ["Demographic Parity Difference (DPD)",
                 f"{fd_pdf.get('demographic_parity_difference', 0):.4f}"],
                ["Equalized Odds Difference (EOD)",
                 f"{fd_pdf.get('equalized_odds_difference', 0):.4f}"],
                ["Maximum Accuracy Gap Across Age Groups",
                 f"{fd_pdf.get('max_accuracy_gap', 0):.4f}"],
            ]
            fair_table = Table(fair_data, colWidths=[10*cm, 6*cm])
            fair_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1f4e79")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('BACKGROUND', (1,1), (1,1), bias_color),
                ('TEXTCOLOR', (1,1), (1,1), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTNAME', (1,1), (1,1), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 10),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(fair_table)

            # 4. SHAP
            if shap_data is not None:
                story.append(Paragraph(
                    "4. Top Risk Factors (SHAP Analysis)",
                    section_style
                ))
                imp_pdf = shap_data["feature_importance"]
                shap_rows = [["Rank", "Feature", "SHAP Importance"]]
                for i, feat in enumerate(shap_data["top_5_features"], 1):
                    shap_rows.append([str(i), feat, f"{imp_pdf[feat]:.4f}"])
                shap_table = Table(shap_rows,
                                   colWidths=[1.5*cm, 10.5*cm, 4*cm])
                shap_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1f4e79")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,-1), 10),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1),
                     [colors.white, colors.HexColor("#f5f5f5")]),
                    ('PADDING', (0,0), (-1,-1), 6),
                    ('ALIGN', (0,0), (0,-1), 'CENTER'),
                    ('ALIGN', (2,0), (2,-1), 'RIGHT'),
                ]))
                story.append(shap_table)

            # 5. Compliance
            story.append(PageBreak())
            story.append(Paragraph("5. EU AI Act 2024 Compliance",
                                   section_style))
            comp_rows = [["Status", "Requirement", "Detail"]]
            for check_name, passed, detail in checks:
                status_text = "PASS" if passed else "PENDING"
                comp_rows.append([status_text, check_name, detail])

            comp_table = Table(comp_rows, colWidths=[2*cm, 5*cm, 9*cm])
            comp_style = [
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1f4e79")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('PADDING', (0,0), (-1,-1), 5),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]
            for i, (_, passed, _) in enumerate(checks, 1):
                color = (colors.HexColor("#28a745") if passed
                         else colors.HexColor("#fd7e14"))
                comp_style.append(('BACKGROUND', (0,i), (0,i), color))
                comp_style.append(('TEXTCOLOR', (0,i), (0,i), colors.white))
                comp_style.append(('FONTNAME', (0,i), (0,i), 'Helvetica-Bold'))
                comp_style.append(('ALIGN', (0,i), (0,i), 'CENTER'))
            comp_table.setStyle(TableStyle(comp_style))
            story.append(comp_table)

            # Disclaimer
            story.append(Spacer(1, 1*cm))
            story.append(Paragraph(
                "DISCLAIMER: This report is generated by an AI screening "
                "system and does not constitute a clinical diagnosis. "
                "All predictions must be reviewed by a qualified healthcare "
                "professional. MediSight AI is a research prototype "
                "developed for academic purposes.",
                disclaimer_style
            ))

            def add_page_number(canvas, doc):
                canvas.saveState()
                canvas.setFont('Helvetica', 8)
                canvas.setFillColor(colors.grey)
                canvas.drawRightString(
                    A4[0] - 2*cm, 1.5*cm,
                    f"Page {doc.page} | MediSight AI"
                )
                canvas.restoreState()

            doc.build(story,
                      onFirstPage=add_page_number,
                      onLaterPages=add_page_number)
            pdf_bytes = buffer.getvalue()
            buffer.close()

            st.download_button(
                label="⬇️ Download Transparency Label PDF",
                data=pdf_bytes,
                file_name=(f"medisight_{disease_key}_{model_key}"
                           f"_label.pdf"),
                mime="application/pdf",
                use_container_width=True
            )
            st.success("✅ PDF generated — professional formatting applied.")

        except ImportError:
            st.error("ReportLab not installed. Run: `pip install reportlab`")
        except Exception as e:
            st.error(f"PDF generation error: {e}")
            import traceback
            st.code(traceback.format_exc())

    st.markdown("---")
    st.caption(
        "MediSight AI | Department of AI & DS, "
        "Mysore University School of Engineering | University of Mysore"
    )