# 🇮🇳 India SME Credit Risk & Growth Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-FF6600?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![RBI Data](https://img.shields.io/badge/Data-Real_RBI_Jan'26-00C9A7?style=flat-square)](https://rbi.org.in)

> **An ML-powered SME credit scoring platform integrating real-world Jan 2026 RBI sectoral credit data.**  
> Built with 2,500 synthetic SME records calibrated to real NPA ratios and macro deployment stats.

---

## 🔗 Live Dashboard

**[INSERT STREAMLIT URL]**

---

## 📌 Problem Statement

India's 63M+ MSME sector faces a ₹20–25 trillion formal credit gap. Traditional models often overlook creditworthy but small enterprises. This platform uses a hybrid approach: **Real RBI Sectoral Calibration** + **ML-based SME Scoring** to identify the lending frontier.

---

## 📊 Key Findings (2,500 SMEs, Real RBI Data)

- 🏆 **Model AUC-ROC: 0.9527** — XGBoost classifier trained on 2,500 rule-derived records
- 🏦 **Real-World Calibration**: IT Services (6.4% default) vs Construction (57.7% default) — 9× risk gap
- 💰 **698 hidden opportunity SMEs** identified using a 3-factor auditable rubric
- 📡 **Live RBI Insight**: Micro & Small credit reached **₹10.3L Crore** in Jan 2026 (29.2% YoY growth)

---

## 🖥️ 7-Page Dashboard Structure

| Page | Title | Visuals |
|------|-------|---------|
| 1 | Executive Overview | KPI cards · Capital tier donut · Industry bar |
| 2 | Geographic Intelligence | State credit rankings · Default heatmap |
| 3 | Sector Risk Analysis | **RBI-calibrated** risk stratification |
| 4 | Company Profile Analysis | Age–Score scatter · Violin · Governance proxies |
| 5 | Hidden Opportunity Finder | 3-factor auditable SME table + state bars |
| 6 | **Model Card** | ROC/PR curves · Threshold analysis · Transparency |
| 7 | **Live RBI Data** | Real macro KPIs · Sub-industry detail · Lineage |

---

## 📁 Project Structure

```
sme_credit_platform/
│
├── data/
│   └── sme_clean.csv               ← 2,500-row SME dataset
│
├── rbi_data/                       ← Real RBI Excel & CSV files
│   ├── rbi_msme_macro.json         ← Jan 2026 macro stats
│   └── rbi_sector_calibration.json ← Blended risk scores
│
├── outputs/
│   ├── feature_importance.png      ← XGBoost feature importance
│   ├── confusion_matrix.png        ← Model confusion matrix
│   └── model_metrics.json          ← AUC-ROC + summary
│
├── fetch_rbi_data.py               ← Downloads real Excel files from RBI
├── parse_rbi_data.py               ← Extracts clean CSVs and calibration JSON
├── module1_data_gen.py             ← Data generation (v2: 2,500 records)
├── module2_model.py                ← XGBoost training
├── app.py                          ← 7-page Streamlit application
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Fetch and Parse Real RBI Data
python fetch_rbi_data.py
python parse_rbi_data.py

# Step 3: Generate SME Data and Train Model
python module1_data_gen.py
python module2_model.py

# Step 4: Run Streamlit App
streamlit run app.py
```

---

## 🎓 Admissions CV Line

> *"Architected an SME Credit Risk Platform for the Indian market, integrating **real Jan 2026 RBI credit data** with a 2,500-record synthetic SME dataset. Developed a **7-page Streamlit dashboard** with XGBoost scoring (AUC: 0.95), an auditable 3-factor 'Hidden Opportunity' rubric, and a transparent Model Card. Calibrated sector risk weights using real-world NPA ratios, identifying 690+ underserved lending opportunities across 21 states."*
