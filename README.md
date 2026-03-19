# 🇮🇳 India SME Credit Risk & Growth Intelligence Platform
### *Phase 6 — Magic UI 2.0 Premium Deployment*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-6366F1?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/UI-Magic_UI_2.0-EC4899?style=flat-square)](https://streamlit.io)
[![Data](https://img.shields.io/badge/Data-Real_RBI_Jan'26-6366F1?style=flat-square)](https://rbi.org.in)

---

## 🌟 The "Magic UI 2.0" Aesthetic
This platform has been meticulously redesigned to match the highest standards of modern fintech design, featuring:

- **Indigo & Pink Palette**: A high-fidelity, high-contrast visual system designed for clarity and impact.
- **Refined Glassmorphism**: Micro-tuned `backdrop-filter` and semi-transparent borders for a deep, professional "frosted glass" feel.
- **Bento Grid Layout**: Asymmetric, responsive card arrangements that prioritize the most critical risk signals.
- **Interactive Credit Scoring**: A real-time "What-If" module with a **Border Beam** animation for a truly premium UX.
- **Transparent Methodology**: Direct access to the **Model Card** (XGBoost) and **RBI Data Lineage**.

---

## 📌 Problem Statement & Technical Methodology
India's 63M+ MSME sector faces a **₹25 trillion formal credit gap**. This platform bridges that gap using:

1. **Real-World Calibration**: Data is calibrated against the **RBI Sectoral Deployment of Bank Credit Reports (Jan 2026)**, representing ₹10.3L Cr in outstanding MSME credit.
2. **Predictive Analytics**: Uses XGBoost to stratify 2,500 SME records across internal risk proxies (Age, Capital, Sector Risk, Management Diversity).
3. **Geographic Intelligence**: State-level risk weightings based on actual bank credit availability and sectoral growth trends.

---

## 📂 Project Architecture & Setup

```bash
sme_credit_platform/
├── app.py                      # Core Premium Dashboard (Phase 6)
├── generate_real_data.py       # RBI-calibrated synthesis engine
├── module2_model.py            # XGBoost training & tuning pipeline
├── fetch_rbi_data.py           # Real-time RBI data scraper
├── parse_rbi_data.py           # RBI PDF/XLSX parser for calibration
├── data/                       # Calibrated datasets (2,500 records)
└── outputs/                    # Model artifacts & metrics
```

### 🚀 Quick Start
1. `pip install -r requirements.txt`
2. `python fetch_rbi_data.py && python parse_rbi_data.py` (Pull latest RBI stats)
3. `python generate_real_data.py` (Synthesize dataset with real weights)
4. `streamlit run app.py`

---

## 🎓 Admissions & Career Highlight
> *"Architected a premium SME Credit Risk Platform for the Indian market. Integrated **real Jan 2026 RBI credit data** into a predictive XGBoost engine. Designed a **high-fidelity 'Magic UI 2.0' dashboard** in Streamlit using custom CSS, achieving a seamless premium experience. Calibrated risk models based on actual state-level sectoral deployment, identifying high-yield lending opportunities across 21 Indian states."*

---
© 2026 | Built for Excellence
