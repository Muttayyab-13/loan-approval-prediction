# Loan Approval Prediction — Design Spec

**Date:** 2026-07-20
**Type:** Beginner-friendly ML classification project (semester submission)

## 1. Purpose

Predict whether a loan application is **Approved** or **Rejected** from applicant
details, using the Kaggle dataset `architsharma01/loan-approval-prediction-dataset`.
Deliver a clean, well-commented ML pipeline plus a Streamlit dashboard that matches
the course "Report Submission Guide" PDF (required visuals, model comparison table,
confusion matrix, UI/dashboard).

Constraints (from the user): simple, readable, minimal abstraction, plain functions
(no classes), short comments per step, **no advanced tuning / no GridSearch**, but
code organized into proper folders/modules.

## 2. Dataset

Kaggle `loan_approval_dataset.csv`, ~4269 rows. Known quirks: **leading spaces in
column names and in string values**. Columns:

- `loan_id` (identifier — dropped)
- `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`
- `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_assets_value`
- `education` (Graduate / Not Graduate) — categorical
- `self_employed` (Yes / No) — categorical
- `loan_status` (Approved / Rejected) — **target (binary)**

The real CSV is supplied by the user in `data/`. A synthetic placeholder with the
same schema is auto-generated if the file is missing, so the pipeline always runs.

## 3. File / module structure

```
ML_Final_Project/
├── data/loan_approval_dataset.csv     # real CSV (synthetic placeholder seeded if missing)
├── src/
│   ├── __init__.py
│   ├── config.py                      # paths (from BASE_DIR), column lists, encoding maps
│   ├── generate_sample_data.py        # creates synthetic placeholder CSV
│   ├── data_preprocessing.py          # load, strip, fill missing, encode, split, scale
│   ├── eda.py                         # Fig 1 class dist, Fig 2 histogram, Fig 3 heatmap
│   ├── models.py                      # build + train the 4 models
│   └── evaluation.py                  # metrics table, Fig 4/5/6, save best model + scaler
├── models/                            # best_model.joblib, scaler.joblib, model_metadata.joblib
├── report_assets/                     # 6 PNGs + model_comparison.csv
├── main.py                            # entry point: runs full pipeline in order
├── app.py                             # Streamlit dashboard
├── requirements.txt
└── README.md
```

`config.py` is the single source of truth (paths + column order + encoding maps),
imported by both the pipeline and `app.py` so the app encodes inputs exactly like
training. `BASE_DIR = Path(__file__).resolve().parent.parent` makes paths work from
any working directory.

## 4. Preprocessing (`data_preprocessing.py`)

1. Load CSV; `df.columns = df.columns.str.strip()`; strip whitespace from string cells.
2. Drop `loan_id`.
3. Missing values: numeric → median, categorical → mode.
4. Encode with explicit maps: `education` {Graduate:1, Not Graduate:0},
   `self_employed` {Yes:1, No:0}, `loan_status` {Approved:1, Rejected:0}.
5. `X` = all features, `y` = `loan_status`.
6. Split 80/20 (`random_state=42`, stratified).
7. `StandardScaler` fit on train, transform train+test.

## 5. EDA (`eda.py`) → `report_assets/`

- **Fig 1** `fig1_target_distribution.png` — bar chart, Approved vs Rejected.
- **Fig 2** `fig2_feature_distribution.png` — histogram of `cibil_score`.
- **Fig 3** `fig3_correlation_heatmap.png` — correlation heatmap of numeric features.

Each figure has a clear title/caption.

## 6. Models (`models.py`)

Simple defaults, no tuning:
- `LogisticRegression(max_iter=1000)`
- `DecisionTreeClassifier(random_state=42)`
- `RandomForestClassifier(random_state=42)`
- `KNeighborsClassifier()`

## 7. Evaluation (`evaluation.py`)

- Metrics per model: Accuracy, Precision, Recall, F1 (positive class = Approved=1).
- Print comparison table (pandas) + save `report_assets/model_comparison.csv`.
- **Fig 4** `fig4_accuracy_comparison.png` — accuracy bar chart.
- **Fig 5** `fig5_precision_recall_f1.png` — grouped Precision/Recall/F1 bar chart.
- **Fig 6** `fig6_confusion_matrix.png` — confusion matrix heatmap for the best model.
- **Best model = highest accuracy.**
- Save `models/best_model.joblib`, `models/scaler.joblib`,
  `models/model_metadata.joblib` (feature order, best model name, metrics dict).

## 8. Dashboard (`app.py`)

- Input form for all 11 applicant fields (numbers + selectboxes for the 2 categoricals).
- On submit: build feature vector in `config` order → encode → scale with loaded
  scaler → predict with loaded model → show **Approved / Rejected** + confidence.
- "Model Performance" section: comparison table (from CSV), accuracy chart (Fig 4),
  confusion matrix (Fig 6).
- Loads artifacts from `models/` and images from `report_assets/`. Shows a friendly
  message if the pipeline hasn't been run yet.

## 9. Run commands

```
python main.py          # trains everything, writes models/ and report_assets/
streamlit run app.py    # launches the dashboard
```

## 10. Out of scope (YAGNI)

No hyperparameter tuning, no cross-validation, no cloud deploy, no auth, no database.
The report/PPT themselves are the user's to write (this covers code + visuals + dashboard).
