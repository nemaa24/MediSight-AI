# MediSight AI: A Transparency Label for Healthcare Prediction Systems

MediSight AI is an automated transparency framework designed to wrap around tabular healthcare prediction models. Inspired by the standardized formatting of a nutrition facts panel, it abstracts complex model metadata into a clean, human-readable transparency label at the point of prediction, allowing clinicians to evaluate model trustworthiness at a glance.

The project is implemented across four distinct disease domains:

- Diabetes
- Heart Disease
- Liver Disease
- Chronic Kidney Disease (CKD)

---

# System Architecture & Pipeline

The system utilizes a decoupled architecture that separates a multi-stage **Offline Processing Pipeline** from a lightweight, high-performance **Online Presentation Layer**.

## 1. Data Layer
- Automated data ingestion
- Data cleaning
- Categorical encoding
- Missing value imputation

Supports heterogeneous clinical and survey datasets.

## 2. Data Preparation
- Feature scaling
- Stratified train-test split
- Class balancing using **SMOTE**

## 3. Modeling Layer

For each disease, three independent classifiers are trained:

- Logistic Regression
- Random Forest
- XGBoost

All models are calibrated using **Platt Scaling** to generate statistically reliable probability estimates.

## 4. Analysis Layer

Generates model transparency metrics including:

- Global SHAP feature importance
- Local SHAP explanations
- Fairlearn-based age disparity analysis

## 5. Presentation Layer

A responsive **Streamlit** dashboard displays:

- Live disease prediction
- Model confidence
- Transparency label
- Explanation metrics

---

# Repository Structure

```text
MediSight-AI/
│
├── audit/
│   └── predictions.jsonl
│
├── dashboard/
│   └── app.py
│
├── datasets/
│   └── processed/
│
├── models/
│
├── notebooks/
│
└── requirements.txt
```

### Directory Description

| Folder | Purpose |
|---------|----------|
| `audit/` | Persistent JSON-Lines prediction audit trail |
| `dashboard/` | Streamlit web application |
| `datasets/` | Raw and processed datasets |
| `models/` | Trained models, scalers, calibration objects and metrics |
| `notebooks/` | Data processing and model development pipeline |
| `requirements.txt` | Python dependency manifest |

---

# Technical Prerequisites

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| Processor | Intel Core i5 (8th Generation or later) or equivalent |
| Memory | Minimum 8 GB RAM (16 GB recommended) |
| Storage | 2 GB available disk space |

> **Note:** The project is fully optimized for CPU execution. A dedicated GPU is not required.

---

## Software Requirements

- Windows 10 / Windows 11
- Ubuntu 20.04+
- macOS
- Python 3.11

### Core Libraries

- scikit-learn
- xgboost
- shap
- fairlearn
- streamlit
- reportlab
- plotly

---

# Installation & Deployment

## 1. Clone the Repository

```bash
git clone https://github.com/nemaa24/medisight-ai.git
cd medisight-ai
```

---

## 2. Initialize Git LFS

The trained machine learning models are stored using **Git Large File Storage (LFS)**.

```bash
git lfs install
git lfs pull
```

> **Important**
>
> If `git lfs pull` is not executed, the model files remain as lightweight pointer files rather than the actual binaries, causing the application to fail during runtime.

---

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The application will automatically open in your default browser.

---

# Core Transparency Disclosures

To align with responsible AI principles, MediSight AI explicitly surfaces several important model limitations.

## Survey-Based Data Provenance

The **Diabetes** and **Heart Disease** models are trained using self-reported CDC survey data rather than Electronic Health Records (EHRs).

These models are intended solely as an academic decision-support prototype.

---

## Custom Classification Threshold

The Heart Disease dataset contains approximately **9% positive cases**, creating severe class imbalance.

Rather than using the default probability threshold of **0.50**, the deployed model uses a calibrated decision threshold of **0.175** to improve screening recall.

---

## Chronic Kidney Disease Separability

All three classifiers achieve **100% evaluation performance** on the CKD dataset.

This does **not** indicate perfect generalization.

Instead, the transparency framework highlights this behavior as a consequence of:

- Strong linear feature separability
- Small validation cohort (80 test samples)

This disclosure is intentionally presented to encourage critical interpretation of performance metrics.

---

# Development Team

This project was developed under the guidance of:

**Dr. Sayeda Umera Almas**  
Assistant Professor  
Department of Artificial Intelligence & Data Science  
Mysore University School of Engineering

### Team Members

- Karan Kumar R
- Nema Nag
- Prajwal A N
- Rakshith S

---

# License

This project is licensed under the **MIT License**.
