"""
evaluation.py — score the models and save the results.

Makes the comparison table and Figures 4-6, then saves the best model + scaler.
  Figure 4: accuracy bar chart
  Figure 5: grouped Precision / Recall / F1 bar chart
  Figure 6: confusion matrix heatmap for the best model
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config import (
    FEATURE_ORDER,
    FIG4_ACCURACY,
    FIG5_PRF,
    FIG6_CONFUSION,
    COMPARISON_CSV,
    BEST_MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
)


def evaluate_models(models, X_test, y_test):
    """Score each model on Accuracy, Precision, Recall, F1. Returns a table (DataFrame)."""
    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        rows.append({
            "Model": name,
            # pos_label=1 means we measure performance for the "Approved" class.
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, preds, pos_label=1, zero_division=0),
            "F1": f1_score(y_test, preds, pos_label=1, zero_division=0),
        })

    results = pd.DataFrame(rows).set_index("Model")
    return results


def save_comparison_table(results):
    """Print the comparison table and save it as a CSV for the report/dashboard."""
    print("\n===== Model Comparison =====")
    print(results.round(3))
    results.round(4).to_csv(COMPARISON_CSV)
    print(f"[evaluation] Saved comparison table to {COMPARISON_CSV}")


def plot_accuracy(results):
    """Figure 4: bar chart comparing accuracy of all models."""
    plt.figure(figsize=(7, 4))
    bars = plt.bar(results.index, results["Accuracy"], color="#4c72b0")
    plt.title("Figure 4: Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    for bar in bars:  # label each bar with its accuracy
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}",
                 ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG4_ACCURACY, dpi=120)
    plt.close()


def plot_prf(results):
    """Figure 5: grouped bar chart comparing Precision, Recall, and F1."""
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(results.index))  # one group per model
    width = 0.25                       # width of each bar

    plt.figure(figsize=(8, 4))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, results[metric], width, label=metric)

    plt.title("Figure 5: Precision, Recall & F1 Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(x + width, results.index, rotation=15)  # center the model labels
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG5_PRF, dpi=120)
    plt.close()


def plot_confusion(best_name, models, X_test, y_test):
    """Figure 6: confusion matrix heatmap for the best model."""
    best_model = models[best_name]
    cm = confusion_matrix(y_test, best_model.predict(X_test))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    plt.title(f"Figure 6: Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIG6_CONFUSION, dpi=120)
    plt.close()


def save_best(models, results, scaler):
    """Pick the best model (highest accuracy) and save it + the scaler + info."""
    best_name = results["Accuracy"].idxmax()
    best_model = models[best_name]

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Metadata helps the dashboard know which model won and its scores.
    metadata = {
        "best_model_name": best_name,
        "feature_order": FEATURE_ORDER,
        "metrics": results.round(4).to_dict(orient="index"),
    }
    joblib.dump(metadata, METADATA_PATH)

    print(f"\n[evaluation] Best model: {best_name}")
    print(f"[evaluation] Saved model, scaler, and metadata to the models/ folder")
    return best_name
