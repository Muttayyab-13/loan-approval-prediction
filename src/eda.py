"""
eda.py — Exploratory Data Analysis charts (Figures 1-3).

Makes three PNGs and saves them in report_assets/:
  Figure 1: target class distribution (Approved vs Rejected)
  Figure 2: distribution of one feature (cibil_score)
  Figure 3: correlation heatmap of all features
"""

import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend so it works without a screen

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import TARGET, FIG1_TARGET, FIG2_FEATURE, FIG3_HEATMAP


def run_eda(df):
    """Create and save the three EDA charts. Expects the cleaned + encoded df."""

    # --- Figure 1: how many Approved vs Rejected loans ---
    counts = df[TARGET].value_counts().sort_index()  # index 0 = Rejected, 1 = Approved
    plt.figure(figsize=(6, 4))
    plt.bar(["Rejected (0)", "Approved (1)"], counts.values,
            color=["#e74c3c", "#2ecc71"])
    plt.title("Figure 1: Loan Status Distribution")
    plt.ylabel("Number of Applications")
    for i, v in enumerate(counts.values):  # write the count on top of each bar
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG1_TARGET, dpi=120)
    plt.close()

    # --- Figure 2: distribution of the cibil_score feature ---
    plt.figure(figsize=(6, 4))
    plt.hist(df["cibil_score"], bins=30, color="#3498db", edgecolor="white")
    plt.title("Figure 2: CIBIL Score Distribution")
    plt.xlabel("CIBIL Score")
    plt.ylabel("Number of Applicants")
    plt.tight_layout()
    plt.savefig(FIG2_FEATURE, dpi=120)
    plt.close()

    # --- Figure 3: correlation heatmap (all columns are numeric now) ---
    plt.figure(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Figure 3: Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG3_HEATMAP, dpi=120)
    plt.close()

    print("[eda] Saved Figure 1, Figure 2, and Figure 3 to report_assets/")
