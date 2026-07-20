"""
generate_sample_data.py — creates a SYNTHETIC placeholder dataset.

Why: the real Kaggle CSV is not included. This makes a fake file with the SAME
columns (and the same "leading space" quirk the real file has) so the whole
pipeline runs end-to-end. Replace data/loan_approval_dataset.csv with the real
Kaggle file when you have it.

Kaggle source: architsharma01/loan-approval-prediction-dataset
"""

import numpy as np
import pandas as pd

from src.config import DATA_PATH


def make_sample_data(n=800):
    """Build a small, learnable, fake loan dataset and save it to DATA_PATH."""
    rng = np.random.default_rng(42)  # fixed seed => same data every run

    # --- Basic applicant numbers ---
    no_of_dependents = rng.integers(0, 6, n)
    income_annum = rng.integers(200_000, 10_000_000, n)
    loan_amount = (income_annum * rng.uniform(1.0, 5.0, n)).astype(int)
    loan_term = rng.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], n)
    cibil_score = rng.integers(300, 900, n)

    # --- Asset values (loosely tied to income) ---
    residential = (income_annum * rng.uniform(0.0, 3.0, n)).astype(int)
    commercial = (income_annum * rng.uniform(0.0, 2.0, n)).astype(int)
    luxury = (income_annum * rng.uniform(0.0, 4.0, n)).astype(int)
    bank = (income_annum * rng.uniform(0.0, 1.5, n)).astype(int)

    # --- Text columns (WITH leading spaces, like the real dataset) ---
    education = rng.choice([" Graduate", " Not Graduate"], n)
    self_employed = rng.choice([" Yes", " No"], n)

    # --- Decide Approved/Rejected with a simple rule + a little randomness ---
    # Good credit score and a loan that is small vs income => more likely approved.
    loan_to_income = loan_amount / income_annum
    score = (cibil_score - 300) / 600 - 0.15 * loan_to_income
    noise = rng.normal(0, 0.1, n)
    approved = (score + noise) > 0.45
    loan_status = np.where(approved, " Approved", " Rejected")

    # --- Assemble with the real column names (note the leading spaces) ---
    df = pd.DataFrame({
        "loan_id": np.arange(1, n + 1),
        " no_of_dependents": no_of_dependents,
        " education": education,
        " self_employed": self_employed,
        " income_annum": income_annum,
        " loan_amount": loan_amount,
        " loan_term": loan_term,
        " cibil_score": cibil_score,
        " residential_assets_value": residential,
        " commercial_assets_value": commercial,
        " luxury_assets_value": luxury,
        " bank_asset_value": bank,
        " loan_status": loan_status,
    })

    df.to_csv(DATA_PATH, index=False)
    print(f"[generate_sample_data] Wrote {n} synthetic rows to {DATA_PATH}")
    return df


if __name__ == "__main__":
    make_sample_data()
