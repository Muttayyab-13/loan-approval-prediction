# 🎓 Presentation & Viva Guide — Loan Approval Prediction

A one-stop crib sheet for presenting the project and answering examiner questions.
Everything here matches the actual code in this repo.

---

## 1. Project in one line

> A machine-learning classifier that predicts whether a bank loan application will be
> **Approved** or **Rejected**, wrapped in a simple Streamlit web dashboard where you can
> type in an applicant's details and get an instant decision.

**Type of problem:** Supervised learning → **Binary Classification** (two outcomes: Approved / Rejected).

---

## 2. Why this project? (the motivation)

- Banks receive thousands of loan applications; manually checking each is slow and inconsistent.
- A model can learn from **past decisions** and instantly flag likely approvals/rejections.
- It shows the **full ML workflow** end-to-end: data → cleaning → EDA → training → evaluation → deployment.

---

## 3. The dataset

| Detail | Value |
|--------|-------|
| Source | Kaggle — *architsharma01/loan-approval-prediction-dataset* |
| Rows | ~4,269 loan applications |
| Columns | 13 (11 features + 1 ID + 1 target) |
| Target | `loan_status` → **Approved / Rejected** |
| Feature types | 9 numeric + 2 categorical (text) |

**Features used (11):** number of dependents, education, self-employed, annual income,
loan amount, loan term, **CIBIL score**, and four asset values (residential, commercial,
luxury, bank).

**Two quirks worth mentioning (examiners like this):**
1. The raw CSV has **leading spaces** in column names and text values (e.g. `" education"`, `" Graduate"`). The code strips them automatically in `load_data()`.
2. The ID column `loan_id` is **dropped** — it carries no predictive information.

---

## 4. The ML workflow (how the pipeline works, step by step)

This is the story to tell during the presentation. `main.py` runs these in order:

1. **Load & clean** (`data_preprocessing.py`)
   - Read CSV → strip whitespace → fill missing values (median for numbers, mode for text) → drop rows with no target.
2. **Encode text → numbers**
   - `Graduate → 1 / Not Graduate → 0`, `Yes → 1 / No → 0`, `Approved → 1 / Rejected → 0`.
3. **EDA — 3 charts** (`eda.py`)
   - Figure 1: Approved vs Rejected counts, Figure 2: CIBIL score histogram, Figure 3: correlation heatmap.
4. **Split & scale** (`split_and_scale()`)
   - **80% train / 20% test**, `stratify=y` to keep the same Approved/Rejected ratio in both.
   - **StandardScaler** fit **only on training data**, then applied to both (prevents data leakage).
5. **Train 4 models** (`models.py`) — with default settings, no tuning.
6. **Evaluate** (`evaluation.py`) — score each on 4 metrics, make Figures 4–6, save the comparison table.
7. **Save the best model** — highest accuracy wins; saved as `best_model.joblib` with the scaler + metadata.
8. **Deploy** (`app.py`) — Streamlit form loads the saved model and predicts on new input.

---

## 5. The four models

All trained with **simple default settings** (no GridSearch / no hyperparameter tuning) — kept intentionally beginner-friendly.

| Model | Idea in one line |
|-------|------------------|
| Logistic Regression | Draws a weighted line between the two classes. |
| Decision Tree | Asks yes/no questions on features to reach a decision. |
| Random Forest | Many decision trees vote; reduces overfitting. |
| K-Nearest Neighbors | Looks at the *k* most similar past applicants. |

---

## 6. Results (know these numbers!)

Scored on the 20% test set. Metrics for the **Approved** class.

| Model | Accuracy | Precision | Recall | F1 |
|-------|:--------:|:---------:|:------:|:---:|
| Logistic Regression | 0.913 | 0.921 | 0.942 | 0.931 |
| **Decision Tree** ⭐ | **0.981** | **0.985** | **0.985** | **0.985** |
| Random Forest | 0.980 | 0.983 | 0.985 | 0.984 |
| K-Nearest Neighbors | 0.895 | 0.906 | 0.927 | 0.916 |

**Winner: Decision Tree (~98% accuracy).**

**Why does the tree win?** `cibil_score` (credit score) is by far the strongest predictor,
and the approval rule in the data is close to a **threshold on the credit score** — exactly
the kind of clean split a decision tree captures perfectly. Tree-based models therefore
score very high here.

---

## 7. The 6 required visualizations

| Figure | What it shows | Made in |
|:------:|---------------|---------|
| 1 | Loan status distribution (Approved vs Rejected) | `eda.py` |
| 2 | CIBIL score distribution (feature histogram) | `eda.py` |
| 3 | Feature correlation heatmap | `eda.py` |
| 4 | Model accuracy comparison (bar chart) | `evaluation.py` |
| 5 | Precision / Recall / F1 comparison (grouped bars) | `evaluation.py` |
| 6 | Confusion matrix of the best model | `evaluation.py` |

All saved as PNGs in `report_assets/`.

---

## 8. What each file does (file-by-file)

| File | Role |
|------|------|
| `main.py` | **Entry point.** Runs the whole pipeline in order (load → EDA → train → evaluate → save). |
| `app.py` | **Streamlit dashboard.** Input form → loads saved model → shows Approved/Rejected + confidence + performance charts. |
| `src/config.py` | **Central settings.** All file paths, column lists, feature order, and text→number maps. Shared by pipeline *and* app so encoding always matches. |
| `src/data_preprocessing.py` | Load, clean (strip spaces, fill missing), encode text, split 80/20, scale. |
| `src/eda.py` | Builds Figures 1–3 (exploratory charts). |
| `src/models.py` | Defines and trains the 4 models. |
| `src/evaluation.py` | Scores models, builds Figures 4–6, saves comparison CSV + best model + scaler + metadata. |
| `src/generate_sample_data.py` | Creates a synthetic placeholder CSV if the real Kaggle file is missing, so the project always runs. |
| `requirements.txt` | Python libraries needed (pandas, scikit-learn, matplotlib, seaborn, streamlit, joblib). |
| `models/` | Saved artifacts: `best_model.joblib`, `scaler.joblib`, `model_metadata.joblib`. |
| `report_assets/` | 6 chart PNGs + `model_comparison.csv`. |
| `README.md` / `SETUP.md` | Project overview / run instructions. |

---

## 9. How to run (live demo commands)

```bash
pip install -r requirements.txt   # 1. install libraries
python main.py                    # 2. train models + make all charts
streamlit run app.py              # 3. open the dashboard (http://localhost:8501)
```

> Always run `python main.py` **before** the dashboard — the app loads the model files that `main.py` creates.

---

## 10. Likely viva questions & ready answers

**Q: Is this classification or regression? Why?**
Classification — the output is a category (Approved/Rejected), not a continuous number.

**Q: Why did you split the data into train and test?**
To measure how the model performs on **unseen** data. Testing on training data would give a misleadingly high score (memorization, not learning).

**Q: What does `stratify=y` do?**
Keeps the same Approved/Rejected ratio in both train and test sets, so the split isn't biased.

**Q: Why scale the features? Why fit the scaler only on training data?**
Features like income (millions) and CIBIL score (300–900) are on very different ranges; scaling puts them on a comparable scale (helps KNN and Logistic Regression). We fit the scaler on **training data only** to avoid **data leakage** — the test set must stay unseen.

**Q: Why is accuracy alone not enough?**
If classes are imbalanced, accuracy can be misleading. That's why we also report **Precision, Recall, and F1**, and show a **confusion matrix**.

**Q: What's the difference between precision and recall here?**
Precision = of loans we *predicted* Approved, how many truly were. Recall = of loans that *truly* were Approved, how many we caught. F1 balances both.

**Q: What is a confusion matrix?**
A 2×2 table of True/False Positives/Negatives — it shows *what kind* of mistakes the model makes (e.g. approving a bad applicant vs rejecting a good one).

**Q: Which feature matters most?**
`cibil_score` (credit score) — visible in the correlation heatmap and it's why tree models score ~98%.

**Q: Why did the Decision Tree win over Logistic Regression?**
The approval rule is close to a threshold split on credit score — trees model such splits naturally, while a linear model is less flexible.

**Q: Did you tune hyperparameters?**
No — models use default settings on purpose to keep it simple and beginner-friendly. Tuning (GridSearchCV) is a clear next step for improvement.

**Q: How does the dashboard use the model?**
It loads `best_model.joblib` + `scaler.joblib`, builds the input row in the **exact same feature order** as training (from `config.py`), scales it with the **same** scaler, and calls `predict()` — this consistency is why `config.py` is shared.

**Q: How do you handle missing values?**
Numeric columns → filled with the column **median**; text columns → filled with the **mode** (most common value); rows with a missing target are dropped.

**Q: What could you improve?**
Hyperparameter tuning, cross-validation, feature importance analysis, handling class imbalance, and trying more models (e.g. XGBoost).

---

## 11. 30-second elevator summary (for the intro)

> "We built a loan approval predictor. We took a real Kaggle dataset of ~4,300 applications,
> cleaned and encoded it, explored it with charts, and trained four classification models —
> Logistic Regression, Decision Tree, Random Forest, and KNN. We compared them on accuracy,
> precision, recall, and F1. The **Decision Tree performed best at ~98% accuracy**, mainly
> because the applicant's **credit score** is the dominant factor. We saved the best model and
> built a **Streamlit dashboard** where anyone can enter applicant details and instantly get an
> Approved or Rejected prediction with a confidence score."
