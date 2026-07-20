# 🏦 Loan Approval Prediction

A simple, beginner-friendly machine learning project that predicts whether a loan
application will be **Approved** or **Rejected**, plus a Streamlit dashboard to try it.

## Dataset

Kaggle: [architsharma01/loan-approval-prediction-dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

- Target column: `loan_status` (Approved / Rejected)
- ~4,269 rows, 13 columns (applicant income, loan amount, CIBIL score, assets, etc.)
- **Quirk:** the real CSV has leading spaces in its column names and text values
  (e.g. `" education"`, `" Graduate"`). The pipeline strips these automatically.

> The real CSV is **not** included. If `data/loan_approval_dataset.csv` is missing,
> `main.py` generates a small **synthetic placeholder** so everything runs. Download
> the real file from Kaggle and drop it in `data/loan_approval_dataset.csv` to use real data.

## Project structure

```
ML_Final_Project/
├── data/loan_approval_dataset.csv   # dataset (real CSV goes here)
├── src/
│   ├── config.py                    # paths, column lists, encoding maps (shared settings)
│   ├── generate_sample_data.py      # makes the synthetic placeholder CSV
│   ├── data_preprocessing.py        # load, clean, encode, split, scale
│   ├── eda.py                       # Figures 1-3 (EDA charts)
│   ├── models.py                    # builds + trains the 4 models
│   └── evaluation.py                # metrics, Figures 4-6, saves best model
├── models/                          # best_model.joblib, scaler.joblib, metadata (generated)
├── report_assets/                   # all 6 charts + model_comparison.csv (generated)
├── main.py                          # runs the full pipeline
├── app.py                           # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Models

Four classification models with simple default settings (no tuning / no GridSearch):

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors

Each is scored on **Accuracy, Precision, Recall, and F1**. The best model (highest
accuracy) is saved for the dashboard.

## Charts (saved in `report_assets/`)

| Figure | Chart |
|--------|-------|
| Figure 1 | Loan status distribution (target classes) |
| Figure 2 | CIBIL score distribution (a feature histogram) |
| Figure 3 | Feature correlation heatmap |
| Figure 4 | Model accuracy comparison |
| Figure 5 | Precision / Recall / F1 comparison |
| Figure 6 | Confusion matrix for the best model |

## How to run

```bash
# 1. Install the dependencies
pip install -r requirements.txt

# 2. Train everything (creates models/ and report_assets/)
python main.py

# 3. Launch the dashboard
streamlit run app.py
```

Run both commands from the project root folder.
