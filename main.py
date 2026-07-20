"""
main.py — run the whole Loan Approval Prediction pipeline in order.

Usage:  python main.py

It will:
  1. Load and clean the data (creates a synthetic sample if the CSV is missing)
  2. Make the EDA charts (Figures 1-3)
  3. Split + scale, then train 4 models
  4. Evaluate them, print a comparison table, save Figures 4-6
  5. Save the best model + scaler for the dashboard
"""

from src.config import DATA_PATH
from src.generate_sample_data import make_sample_data
from src.data_preprocessing import load_data, clean_and_encode, split_and_scale
from src.eda import run_eda
from src.models import build_models, train_models
from src.evaluation import (
    evaluate_models,
    save_comparison_table,
    plot_accuracy,
    plot_prf,
    plot_confusion,
    save_best,
)


def main():
    print("===== Loan Approval Prediction Pipeline =====\n")

    # Step 0: make sure we have a dataset. If not, create a synthetic placeholder.
    if not DATA_PATH.exists():
        print("[main] No dataset found. Creating a SYNTHETIC placeholder.")
        print("[main] Replace data/loan_approval_dataset.csv with the real Kaggle file.\n")
        make_sample_data()

    # Step 1: load and prepare the data.
    df = load_data()
    df = clean_and_encode(df)
    print(f"[main] Data ready: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Step 2: exploratory charts.
    run_eda(df)

    # Step 3: split, scale, and train the models.
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    models = train_models(build_models(), X_train, y_train)

    # Step 4: evaluate and make the result charts.
    results = evaluate_models(models, X_test, y_test)
    save_comparison_table(results)
    plot_accuracy(results)
    plot_prf(results)

    # Step 5: find the best model, draw its confusion matrix, and save everything.
    best_name = results["Accuracy"].idxmax()
    plot_confusion(best_name, models, X_test, y_test)
    save_best(models, results, scaler)

    print("\n===== Done! Charts are in report_assets/, model is in models/ =====")
    print("Next: run  streamlit run app.py")


if __name__ == "__main__":
    main()
