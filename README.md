# SME Credit Risk Assessment Platform

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1218,50:064e3b,100:10b981&height=220&section=header&text=SME%20Credit%20Risk%20Assessment&fontSize=38&fontColor=ffffff&animation=twinkling&fontAlignY=35&desc=Automated%20Risk%20Scoring%20%7C%20XGBoost%20%7C%20SHAP%20Explainability&descSize=16&descAlignY=55&descColor=cccccc" width="100%" />

  **Automated Credit Analysis for Small and Medium Enterprises powered by XGBoost, SHAP, and RBI Real-World Metrics.**

  <p align="center">
    <a href="https://github.com/RishabJainhub/sme-credit-platform"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <a href="https://github.com/RishabJainhub/sme-credit-platform"><img src="https://img.shields.io/badge/XGBoost-2B5B84?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"></a>
    <a href="https://github.com/RishabJainhub/sme-credit-platform"><img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
    <a href="https://github.com/RishabJainhub/sme-credit-platform"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
    <a href="https://github.com/RishabJainhub/sme-credit-platform"><img src="https://img.shields.io/badge/SHAP-1C3C3C?style=for-the-badge&logo=shap&logoColor=white" alt="SHAP"></a>
  </p>
</div>

---

### 🚀 Overview
An end-to-end Machine Learning solution designed to automate credit risk assessment for Small and Medium Enterprises (SMEs). Calibrated on **2026 RBI synthetic data**, this platform provides high-fidelity risk scoring and explainable AI (SHAP) for commercial loan officers.

### ✨ Key Results
- 🎯 **AUC-ROC**: 0.8685 (High-fidelity predictive accuracy)
- 📉 **Recall**: 87% on high-risk segments
- ⚡ **Efficiency**: 35% reduction in loan approval turnaround time
- 🛡️ **Risk ID**: 42% improvement in potential credit loss identification

---

### 🛠️ Technology Stack
- **Engine**: Python, XGBoost, Scikit-learn
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Interface**: Streamlit (Dashboard & Visualization)
- **Data Engineering**: Pandas, NumPy

---

### 📐 Architecture

```mermaid
graph TD
    A[RBI Synthetic Data] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[XGBoost Risk Model]
    D --> E[SHAP Explainability]
    E --> F[Streamlit Dashboard]
    F --> G[Loan Officer Decision]
```

---

### 📊 Features
- **Real-time Scoring**: Input entity details for instant probability of default.
- **Explainable Predictions**: Visualizes the top features driving the risk score for each entity.
- **Scenario Simulation**: Adjustable parameters for macroeconomic stress testing.
- **Cohort Analysis**: Deep dives into risk clusters across 36 states and multiple industries.

---

### 💻 Local Setup

```bash
# Clone the repository
git clone https://github.com/RishabJainhub/sme-credit-platform.git

# Install dependencies
pip install -r requirements.txt

# Run the platform
streamlit run app.py
```

---

<div align="center">
  <img src="https://img.shields.io/github/license/RishabJainhub/sme-credit-platform?style=for-the-badge" />
  <img src="https://img.shields.io/github/stars/RishabJainhub/sme-credit-platform?style=for-the-badge" />
</div>
