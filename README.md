# MediSight AI: A Transparency Label for Healthcare Prediction Systems

MediSight AI is an automated transparency framework designed to wrap around tabular healthcare prediction models. Inspired by the standardized formatting of a nutrition facts panel, it abstracts complex metadata into a clean, human-readable transparency label at the point of prediction, allowing clinicians to evaluate model trustworthiness at a glance.

The project is implemented across four distinct disease domains: diabetes, heart disease, liver disease, and chronic kidney disease.

## System Architecture & Pipeline

The system utilizes a decoupled architecture separating a multi-stage **Offline Processing Pipeline** from a thin, high-performance **Online Presentation Layer**.

1. **Data Layer:** Automates ingestion, cleaning, encoding, and missing-value imputation for heterogeneous clinical and survey data.
2. **Data Preparation:** Handles feature scaling, stratified train/test partitioning, and training-set class balancing using SMOTE.
3. **Modeling Layer:** Trains three independent classifiers per disease (Logistic Regression, Random Forest, and XGBoost) and applies Platt Scaling calibration to output statistically reliable probabilities.
4. **Analysis Layer:** Generates global and local feature attributions using SHAP and calculates algorithmic age-disparity metrics via Fairlearn.
5. **Presentation Layer:** A responsive web application that displays the localized transparency label alongside live predictions.


## 📂 Repository Structure

```text
MediSight-AI/
├── audit/                  # Persistent JSON-Lines clinical prediction audit trail
│   └── predictions.jsonl   
├── dashboard/              # Streamlit application source and UI configuration
│   └── app.py              
├── datasets/               # Secure data directory partitioning raw and processed states
│   └── processed/          
├── models/                 # Serialized model pickles, data scalers, and performance metrics
├── notebooks/              # Sequentially structured evaluation and processing pipeline
└── requirements.txt        # Exact Python framework dependency manifest

# Technical Prerequisites
Hardware Requirements
Processor: Intel Core i5 (8th Generation or later) or equivalent.

Memory: 8 GB RAM minimum (16 GB highly recommended to support large ensemble binary loads).

Storage: 2 GB of available local storage space.

Note: Execution is entirely optimized for CPU; dedicated graphics hardware is not required.

Software Profile
Operating System: Windows 10/11, Ubuntu 20.04+, or macOS.

Environment: Python 3.11.

Core Libraries: scikit-learn, xgboost, shap, fairlearn, streamlit, reportlab, plotly.

## Installation & Deployment
1. Clone the Repository and Initialize Git LFS
Because this project tracks compressed machine learning binaries, Git Large File Storage (LFS) must be initialized explicitly to pull down the underlying files:

# Clone the project source
git clone [https://github.com/nemaa24/medisight-ai.git](https://github.com/nemaa24/medisight-ai.git)
cd medisight-ai

# Fetch the tracking pointers for the model binaries
git lfs install
git lfs pull

## Important: Failing to run git lfs pull leaves the model files as empty text pointer stubs, which will cause the execution layer to fail at runtime.

2. Set Up the Virtual Environment
Isolate your environment and install the required dependencies:

python -m venv venv
source venv/bin/activate  # Windows users use: venv\Scripts\activate
pip install -r requirements.txt

3. Run the Dashboard
Execute the server deployment directly from the repository root:

streamlit run dashboard/app.py

## Core Disclosures & Technical Limitations
In accordance with strict AI transparency objectives, the system surfaces and explicitly documents its internal data and evaluation limits:

Survey-Based Provenance: The Diabetes and Heart models are trained on self-reported population survey data from the CDC rather than direct clinical electronic health records (EHR). The platform serves as an academic, decision-support prototype.

Custom Decision Thresholds: Due to severe class imbalance within the Heart dataset (9% positive baseline), the platform optimizes screening recall by overriding the default 0.5 classification cut with a fine-tuned threshold of 0.175.

The Kidney Separability Paradox: All classifiers trained on the Chronic Kidney Disease dataset return perfect 1.000 performance metrics across the board. This behavior is highlighted within the framework as a characteristic of linear feature separability over a tiny validation cohort (80 test records), rather than flawless model generalization.

## Development Team
This framework was developed under the guidance of Dr. Sayeda Umera Almas (Asst. Professor, Dept. of AI&DS, Mysore University School of Engineering).

Karan Kumar R

Nema Nag

Prajwal A N

Rakshith S

Licensed under the MIT License.

