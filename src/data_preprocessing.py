"""
data_preprocessing.py — load the CSV and get it ready for modeling.

Steps: load -> strip whitespace -> handle missing values -> encode text columns
-> split into train/test -> scale the features.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PATH,
    DROP_COLS,
    TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    EDUCATION_MAP,
    SELF_EMPLOYED_MAP,
    TARGET_MAP,
)


def load_data():
    """Read the CSV and strip the extra whitespace found in this dataset."""
    df = pd.read_csv(DATA_PATH)

    # 1) Clean the column names (the Kaggle file has leading spaces like " education").
    df.columns = df.columns.str.strip()

    # 2) Clean the text values inside every text column (" Graduate" -> "Graduate").
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def clean_and_encode(df):
    """Drop the ID column, fill missing values, and turn text into numbers."""
    # Drop columns we do not use (just the loan_id here).
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Fill any missing numeric values with that column's median.
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())

    # Fill any missing text values with that column's most common value (mode).
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode()[0])

    # If the target itself is missing, we cannot use that row — drop it.
    df = df.dropna(subset=[TARGET])

    # Encode text columns into numbers using the maps from config.py.
    df["education"] = df["education"].map(EDUCATION_MAP)
    df["self_employed"] = df["self_employed"].map(SELF_EMPLOYED_MAP)
    df[TARGET] = df[TARGET].map(TARGET_MAP)

    return df


def split_and_scale(df):
    """Split 80/20 and scale the features so they are on a similar range."""
    X = df[FEATURE_ORDER]   # features, in the fixed order from config
    y = df[TARGET]          # target (0 = Rejected, 1 = Approved)

    # 80% train, 20% test. stratify=y keeps the same Approved/Rejected ratio in both.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale: fit ONLY on training data, then apply the same scaling to both.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
