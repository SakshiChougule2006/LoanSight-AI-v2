# 💳 LoanSight AI v2 — Advanced Loan Default Prediction

> End-to-end ML system with SHAP explainability, Optuna hyperparameter tuning, SMOTE class balancing, PDF letter generation, batch prediction, what-if simulator, and a full REST API — deployed via Flask.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![Optuna](https://img.shields.io/badge/Optuna-Tuned-yellow)
![SMOTE](https://img.shields.io/badge/SMOTE-Balanced-red)

---

## 🚀 Features

| Feature | Description |
|---|---|
| ⚡ **Live Prediction** | Real-time default risk scoring with animated gauge |
| 🧠 **SHAP Explainability** | See exactly which features push risk up or down |
| 🎛 **What-If Simulator** | Sliders update risk score in real-time |
| ⚖ **Applicant Comparison** | Side-by-side risk comparison of two applicants |
| 📂 **Batch Prediction** | Upload CSV → download predictions for all rows |
| 📄 **PDF Letter Generator** | Auto-generate approval/rejection letters |
| 🔌 **REST API** | 4 documented endpoints |
| 📊 **Analytics Dashboard** | ROC, SHAP summary, feature importance, Optuna chart |

---

## 🧠 ML Pipeline

```
Raw Data (50,000 rows, Lending Club schema)
    ↓
Label Encoding (grade, purpose, home_ownership...)
    ↓
Train/Test Split (80/20, stratified)
    ↓
SMOTE Oversampling (balance classes)
    ↓
Baseline Models (LR, RF, XGBoost)
    ↓
Optuna Tuning (25 trials, XGBoost)
    ↓
SHAP Explainer (TreeExplainer)
    ↓
Flask Deployment (4 API endpoints)
```

---

## 📁 Project Structure

```
loansight_v2/
├── app.py                        # Flask app — 4 API endpoints
├── requirements.txt
├── README.md
├── lending_club_sample.csv       # 50k row Lending Club schema dataset
├── models/
│   ├── xgb_tuned.pkl             # Optuna-tuned XGBoost model
│   ├── scaler_v2.pkl             # StandardScaler
│   ├── shap_explainer.pkl        # TreeExplainer
│   ├── le_map.pkl                # LabelEncoders
│   └── results_v2.json           # Model metrics + best params
├── templates/
│   └── index.html                # Full 6-tab dashboard
└── static/
    └── images/
        ├── shap_summary.png
        ├── roc_v2.png
        ├── optuna_chart.png
        ├── feature_importance_v2.png
        └── correlation_v2.png
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/LoanSight-AI-v2.git
cd LoanSight-AI-v2
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:5000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Single prediction + SHAP explanation |
| POST | `/api/generate_letter` | PDF approval/rejection letter |
| POST | `/api/batch_predict` | Batch CSV prediction |
| POST | `/api/whatif` | What-if scenario scoring |

---

## 📊 Dataset

Synthetic dataset (50,000 rows) generated to match **Lending Club's schema** — same 20 features, same value ranges and distributions as the public Kaggle Lending Club dataset.

| Feature | Description |
|---|---|
| `loan_amnt` | Loan amount requested |
| `int_rate` | Interest rate |
| `grade` | LC assigned loan grade (A–G) |
| `fico_range_low` | FICO credit score (low) |
| `dti` | Debt-to-income ratio |
| `annual_inc` | Annual income |
| `delinq_2yrs` | Delinquencies in last 2 years |
| `revol_util` | Revolving line utilization |
| `purpose` | Loan purpose |
| + 11 more | See lending_club_sample.csv |

---

## 🛠 Tech Stack

- **ML:** XGBoost, Scikit-learn, SHAP, Optuna, imbalanced-learn (SMOTE)
- **Data:** Pandas, NumPy
- **Viz:** Matplotlib, Seaborn
- **Backend:** Flask
- **PDF:** ReportLab
- **Frontend:** Vanilla JS, HTML5/CSS3

---

*Built by Sakshi — AI & ML Engineer | B.Tech CSE, Rajarambapu Institute of Technology*
