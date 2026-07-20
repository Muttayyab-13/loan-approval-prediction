# ⚙️ Setup Guide — Loan Approval Prediction

A quick guide to run the project, understand the models, and know what each file does.

---

## 🚀 How to run

```bash
# 1. (once) create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Train the models + generate all charts (writes to models/ and report_assets/)
python main.py

# 3. Launch the dashboard (opens at http://localhost:8501)
streamlit run app.py
```

Run every command from the **project root** folder. Always run `python main.py` **before**
`streamlit run app.py`, because the dashboard loads the model files that `main.py` creates.

---

## 🤖 Models (important details)

- Four models are trained with **simple default settings** (no tuning / no GridSearch):
  **Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors**.
- Each is scored on **Accuracy, Precision, Recall, F1** (positive class = *Approved*).
- The **best model = highest accuracy**; it is saved to `models/best_model.joblib`.
- On the real Kaggle dataset (4,269 rows), the **Decision Tree wins (~98% accuracy)**.
  `cibil_score` is the strongest predictor, so tree-based models perform very well.

---

## 📂 Folder structure

```
ML_Final_Project/
├── data/            # the dataset (loan_approval_dataset.csv)
├── src/             # all pipeline code (config + preprocessing + eda + models + evaluation)
├── models/          # saved best model, scaler, and metadata (created by main.py)
├── report_assets/   # 6 chart PNGs + model_comparison.csv (created by main.py)
├── docs/            # design spec + implementation plan
├── main.py          # runs the whole pipeline
├── app.py           # Streamlit dashboard
├── requirements.txt # dependencies
├── README.md        # project overview
└── SETUP.md         # this file
```

---

## 📄 What each file does (in short)

| File | What it does |
|------|--------------|
| `main.py` | Entry point that runs the full pipeline in order. Calls every `src/` module from load to save. |
| `app.py` | Streamlit dashboard with an input form. Loads the saved model and shows the prediction + charts. |
| `src/config.py` | One place for all settings: file paths, column names, encoding maps. Shared by the pipeline and app. |
| `src/generate_sample_data.py` | Creates a fake placeholder CSV if the real one is missing. Lets the pipeline run without the Kaggle file. |
| `src/data_preprocessing.py` | Loads and cleans the data. Strips whitespace, fills missing values, encodes text, splits, and scales. |
| `src/eda.py` | Makes the 3 EDA charts (Figures 1–3). Target distribution, a feature histogram, and a correlation heatmap. |
| `src/models.py` | Defines the 4 models and trains them. Returns the trained models to be evaluated. |
| `src/evaluation.py` | Scores the models and makes Figures 4–6. Saves the comparison table and the best model + scaler. |
| `requirements.txt` | Lists the Python libraries to install. Used by `pip install -r requirements.txt`. |

---

## 📊 Generated charts (in `report_assets/`)

| Figure | Chart |
|--------|-------|
| Figure 1 | Loan status distribution (Approved vs Rejected) |
| Figure 2 | CIBIL score distribution (feature histogram) |
| Figure 3 | Feature correlation heatmap |
| Figure 4 | Model accuracy comparison |
| Figure 5 | Precision / Recall / F1 comparison |
| Figure 6 | Confusion matrix for the best model |

---

## 📝 Important notes

- **Dataset quirk:** the real Kaggle CSV has leading spaces in column names and text values
  (e.g. `" education"`, `" Graduate"`). The code strips these automatically.
- **Column name:** the dataset uses `bank_asset_value` (singular), while the others are
  `_assets_value` (plural). The code already matches the real names.
- To use a different dataset version, just replace `data/loan_approval_dataset.csv` and
  re-run `python main.py`. All charts and the saved model update automatically.
