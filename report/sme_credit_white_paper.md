# India SME Credit Risk & Growth Intelligence Platform
## White Paper — Hybrid Research Prototype (Real RBI Data + Synthetic ML)

> **Authors:** [Your Name]
> **Institution:** [University / Programme]
> **Date:** March 2026
> **Classification:** Academic Portfolio Project — SSRN-Ready Prototype
> **Data Source:** Hybrid (Real RBI Sectoral Deployment Jan 2026 + 2,500 Synthetic SME Records)

---

## Abstract

India's MSME sector contributes ~30% of GDP but faces a ₹20–25 trillion credit gap. This paper presents an intelligence platform that bridges real-world macro data with granular ML-based scoring. By integrating **real Jan 2026 RBI sectoral deployment data** with a 2,500-record SME dataset, this platform trains an XGBoost classifier (AUC-ROC: **0.9527**) and identifies **698 creditworthy-but-underserved lending opportunities**. The model uses a hybrid calibration (40% RBI signal + 60% expert prior) to address structural sector risk, proving that data-driven underwriting can unlock significant high-yield SME cohorts currently invisible to traditional lenders.

---

## 1. Executive Summary

- **Records:** 2,500 synthetic Indian SMEs (upgraded from 750 for statistical robustness)
- **Real-Data Anchor:** Jan 2026 RBI Sectoral Deployment (Micro & Small credit: **₹10.3L Cr**)
- **Risk Stratification:** 9× gap in default rates between safest (IT) and riskiest (Construction) sectors
- **ML Performance:** XGBoost AUC-ROC of 0.9527; Model Card provides full PR curve and threshold transparency
- **Addressable Alpha:** 698 "Hidden Opportunity" SMEs identified via an auditable 3-factor rubric

---

## 2. Methodology & Data Architecture

### 2.1 Hybrid Data Generation (v2)
To move beyond generic synthetic data, the v2 generator incorporates:
1. **Tiered NPA Penalties:** Risk scores are adjusted by real-world NPA tiers (RBI proxy), creating a 21-point credit score gap between sectors.
2. **Log-Normal Capitalization:** Authorized capital mimics the real-world power-law distribution of Indian private limited companies.
3. **Governance Proxies:** Director count is weighted as a scoring bonus (2+ directors = higher governance stability).

### 2.2 Real RBI Integration
The platform fetches live Excel files from `rbi.org.in` covering:
- **Statement 1:** Deployment by major sector (Agriculture, Industry, Services).
- **Statement 2:** Industry-wise breakdown (Textiles, Chemicals, Construction).
Sector risk weights in the ML model are **blended** (40% RBI credit growth signal + 60% expert ordering) to ensure the model reflects current macro constraints.

### 2.3 Machine Learning Pipeline
- **Algorithm:** XGBoost Binary Classifier.
- **Top Feature:** Sector Risk Score (Calibrated by RBI).
- **Transparency:** A dedicated **Model Card** (Page 6) documents precision/recall at various decision thresholds (0.30 to 0.80), allowing lenders to adjust based on risk appetite.

---

## 3. Key Findings

### 3.1 The 9× Sector Risk Gap
Under RBI-calibrated parameters, default rates diverge sharply:
- **IT Services:** 6.4% expected default (Score: 69.2)
- **Construction:** 57.7% expected default (Score: 47.9)
This divergence justifies highly differentiated liquidity-based underwriting rather than flat scorecarding.

### 3.2 Live RBI Insight (Jan 2026)
Real sectoral deployment shows **Micro & Small credit growing at 29.2% YoY**, significantly outpacing Large Industry (~4% growth). This "Micro-Momentum" represents a major lending frontier for Fintech/NBFC players.

### 3.3 The Hidden Opportunity Cohort
Using the 3-factor rubric (**Score ≥ 65** AND **Cap < ₹50L** AND **Low-Risk Sector**), the platform surfaces **698 SMEs** representing the "high-yield safe" frontier.

---

## 4. Recommendations & Roadmap

- **For Admissions/Recruiters:** This project demonstrates a complete ETL + ML + BI lifecycle with actual RBI data fetching, proving readiness for Advanced Analytics or FinTech Strategy roles.
- **Production Roadmap:** Replace synthetic SME profiles with GSTN-verified revenue data (OCEN API) and actual CIBIL bureau scores to move from prototype to production.

---

*This white paper is an academic portfolio prototype. For research enquiries, contact the author via LinkedIn.*
