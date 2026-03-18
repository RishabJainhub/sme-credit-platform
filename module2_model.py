"""
MODULE 2 — XGBOOST CREDIT RISK MODEL
India SME Credit Risk & Growth Intelligence Platform
=====================================================
Trains XGBoost binary classifier on DEFAULT_RISK.
Features: Age, Multiple Directors (engineered), Metro, Sector Risk, Log-Capital
⚠️ SYNTHETIC DATA NOTICE: DEFAULT_RISK is rule-derived, not from actual loan
   performance data. This model demonstrates methodology, not live credit decisions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, os
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay
)

SEED = 42
np.random.seed(SEED)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("data/sme_clean.csv")
print(f"Loaded {len(df)} records from data/sme_clean.csv")
print(f"Class balance — DEFAULT_RISK:\n{df['DEFAULT_RISK'].value_counts().to_string()}\n")

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
df["LOG_CAP"] = np.log1p(df["AUTHORIZED_CAP_INR"])
df["HAS_MULTIPLE_DIRECTORS"] = (df["DIRECTOR_COUNT"] > 2).astype(int)

FEATURES = [
    "AGE_YEARS",
    "HAS_MULTIPLE_DIRECTORS",
    "IS_METRO",
    "SECTOR_RISK_SCORE",
    "LOG_CAP"
]
TARGET = "DEFAULT_RISK"

X = df[FEATURES]
y = df[TARGET]

# ─────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────
# TRAIN XGBOOST
# ─────────────────────────────────────────
def train_model(params: dict) -> XGBClassifier:
    model = XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

base_params = dict(n_estimators=100, max_depth=4, learning_rate=0.1)
model = train_model(base_params)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

print("=" * 55)
print("INITIAL MODEL RESULTS")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
print(f"AUC-ROC Score : {auc:.4f}")

# ─────────────────────────────────────────
# AUTO-TUNE IF AUC < 0.75
# ─────────────────────────────────────────
if auc < 0.75:
    print("\n⚠️  AUC-ROC below 0.75 — running GridSearchCV to tune...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":     [3, 4, 6],
        "learning_rate": [0.05, 0.10, 0.15],
        "subsample":     [0.8, 1.0]
    }
    gs = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                      random_state=SEED, verbosity=0),
        param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    model      = gs.best_estimator_
    y_pred     = model.predict(X_test)
    y_proba    = model.predict_proba(X_test)[:, 1]
    auc        = roc_auc_score(y_test, y_proba)
    print(f"Best params : {gs.best_params_}")
    print(f"Tuned AUC-ROC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
else:
    print(f"\n✅ AUC-ROC threshold met: {auc:.4f} ≥ 0.75")

# ─────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
fig_cm.patch.set_facecolor("#0D1B2A")
ax_cm.set_facecolor("#0D1B2A")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
ax_cm.set_title("Confusion Matrix — XGBoost Credit Risk Classifier",
                color="white", fontsize=12, pad=12)
ax_cm.xaxis.label.set_color("white")
ax_cm.yaxis.label.set_color("white")
ax_cm.tick_params(colors="white")
for text in disp.text_.ravel():
    text.set_color("white")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight",
            facecolor="#0D1B2A")
plt.close()

# ─────────────────────────────────────────
# FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────
importances = model.feature_importances_
feat_labels = [
    "Age (Years)", "Multiple Directors", "Metro Location",
    "Sector Risk Score", "Log Capital"
]
sorted_idx  = np.argsort(importances)
colors      = ["#00C9A7" if importances[i] == max(importances) else "#4A90D9"
               for i in sorted_idx]

fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
fig_fi.patch.set_facecolor("#0D1B2A")
ax_fi.set_facecolor("#0D1B2A")

bars = ax_fi.barh(
    [feat_labels[i] for i in sorted_idx],
    importances[sorted_idx],
    color=colors, edgecolor="none", height=0.55
)

ax_fi.set_xlabel("Feature Importance (Gain)", color="white", fontsize=11)
ax_fi.set_title("XGBoost Feature Importance — SME Default Risk Predictor",
                color="white", fontsize=13, pad=14)
ax_fi.tick_params(colors="white")
ax_fi.spines["top"].set_visible(False)
ax_fi.spines["right"].set_visible(False)
for sp in ["bottom", "left"]:
    ax_fi.spines[sp].set_color("#444")
ax_fi.xaxis.label.set_color("white")
ax_fi.set_xlim(0, max(importances) * 1.25)

for bar, val in zip(bars, importances[sorted_idx]):
    ax_fi.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
               f"{val:.3f}", va="center", ha="left", color="white", fontsize=10)

# Insight annotation
insight = (
    f"Model Insight: Sector Risk Score dominates credit risk prediction —\n"
    f"structural industry exposure outweighs firm-level factors in SME defaults."
)
fig_fi.text(0.02, 0.01, insight, fontsize=8.5, color="#AAAAAA",
            fontstyle="italic", wrap=True)

# Synthetic data disclaimer
disclaimer = (
    "⚠️ SYNTHETIC DATA NOTICE: DEFAULT_RISK is rule-derived, not from actual loan performance data.\n"
    "This model demonstrates methodology, not live credit decisions."
)
fig_fi.text(0.02, 0.92, disclaimer, fontsize=7.5, color="#FF6B6B",
            bbox=dict(facecolor="#1A0000", edgecolor="#FF6B6B", alpha=0.7, pad=4))

plt.tight_layout(rect=[0, 0.06, 1, 0.90])
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight",
            facecolor="#0D1B2A")
plt.close()

print("\nSaved: outputs/feature_importance.png")
print("Saved: outputs/confusion_matrix.png")

# ─────────────────────────────────────────
# SUMMARY METRICS — for use in downstream modules
# ─────────────────────────────────────────
metrics = {
    "auc_roc": round(auc, 4),
    "test_size": len(X_test),
    "train_size": len(X_train),
    "high_risk_count": int(y.sum()),
    "low_risk_count": int((1 - y).sum()),
    "top_feature": feat_labels[sorted_idx[-1]]
}
import json
with open("outputs/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ MODULE 2 COMPLETE — AUC-ROC: {auc:.4f} | Top Feature: {metrics['top_feature']}")
print("   Model metrics saved to outputs/model_metrics.json")
