# Loan Approval Prediction Implementation Plan

> **For agentic workers:** Implement task-by-task in build order. This is a simple,
> beginner-friendly ML project — no test suite, no TDD, no git commits (not a repo).
> Verification = running `main.py` end-to-end and checking the generated artifacts.

**Goal:** A clean, modular, well-commented ML pipeline that predicts loan approval and
a Streamlit dashboard, matching the course PDF's required visuals and comparison table.

**Architecture:** Flat functions organized into `src/` modules by responsibility, driven
by a single `main.py` entry point. `config.py` is the shared source of truth (paths,
column order, encoding maps) imported by both the pipeline and `app.py`.

**Tech Stack:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit.

---

## File build order

1. `requirements.txt` — dependencies.
2. `src/__init__.py` — empty, makes `src` a package.
3. `src/config.py` — paths (from BASE_DIR), column lists, encoding maps, figure filenames.
4. `src/generate_sample_data.py` — synthetic placeholder CSV generator.
5. `src/data_preprocessing.py` — load → clean → fill → encode → split → scale.
6. `src/eda.py` — Fig 1/2/3.
7. `src/models.py` — build + train the 4 models.
8. `src/evaluation.py` — metrics table, Fig 4/5/6, save best model/scaler/metadata.
9. `main.py` — orchestrate the pipeline.
10. `app.py` — Streamlit dashboard.
11. `README.md` — run instructions.
12. Run + verify.

---

### Task 1: `requirements.txt`

Pinned-ish, current stable versions:
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
streamlit>=1.30
```

---

### Task 2: `src/config.py`

Single source of truth. Contents:
- `BASE_DIR = Path(__file__).resolve().parent.parent`
- `DATA_DIR`, `MODELS_DIR`, `ASSETS_DIR` (created with `mkdir(parents, exist_ok)`).
- `DATA_PATH = DATA_DIR / "loan_approval_dataset.csv"`.
- `TARGET = "loan_status"`, `DROP_COLS = ["loan_id"]`.
- `NUMERIC_FEATURES` = the 9 numeric cols; `CATEGORICAL_FEATURES = ["education","self_employed"]`.
- `FEATURE_ORDER` = exact column order the model is trained on (numeric + encoded categoricals).
- Encoding maps: `EDUCATION_MAP={"Graduate":1,"Not Graduate":0}`,
  `SELF_EMPLOYED_MAP={"Yes":1,"No":0}`, `TARGET_MAP={"Approved":1,"Rejected":0}`.
- Artifact paths: `BEST_MODEL_PATH`, `SCALER_PATH`, `METADATA_PATH`, `COMPARISON_CSV`.
- Figure paths: `FIG1..FIG6` filenames under `ASSETS_DIR`.

---

### Task 3: `src/generate_sample_data.py`

`make_sample_data(n=800)`: build a synthetic DataFrame with the real schema **and the
dataset's leading-space quirk** (space-prefixed column names + string values), so cleaning
is exercised. Use `numpy` default_rng(42) for reproducibility. Make `loan_status` loosely
depend on `cibil_score` + income/loan ratio so models learn something. Write to `DATA_PATH`.
`if __name__ == "__main__": make_sample_data()`.

---

### Task 4: `src/data_preprocessing.py`

Functions:
- `load_data()` → read `DATA_PATH`, strip column names, strip whitespace from object cols, return df.
- `clean_and_encode(df)` → drop `DROP_COLS`; fill numeric NaN with median, categorical NaN with mode; map education/self_employed/target via config maps; return df.
- `split_and_scale(df)` → `X = df[FEATURE_ORDER]`, `y = df[TARGET]`; `train_test_split(0.2, random_state=42, stratify=y)`; `StandardScaler` fit on train; return `X_train_s, X_test_s, y_train, y_test, scaler`.

---

### Task 5: `src/eda.py`

`run_eda(df)` (takes cleaned+encoded df): saves
- Fig 1: `value_counts` of target as bar chart, title "Figure 1: Loan Status Distribution".
- Fig 2: histogram of `cibil_score`, title "Figure 2: CIBIL Score Distribution".
- Fig 3: `df.corr()` seaborn heatmap, title "Figure 3: Feature Correlation Heatmap".
Each: `plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()`.

---

### Task 6: `src/models.py`

- `build_models()` → dict `{"Logistic Regression": LogisticRegression(max_iter=1000), "Decision Tree": DecisionTreeClassifier(random_state=42), "Random Forest": RandomForestClassifier(random_state=42), "K-Nearest Neighbors": KNeighborsClassifier()}`.
- `train_models(models, X_train, y_train)` → fit each in place, return the dict.

---

### Task 7: `src/evaluation.py`

- `evaluate_models(models, X_test, y_test)` → for each: predict, compute accuracy/precision/recall/f1 (pos_label=1), return a pandas DataFrame (rows=models, cols=metrics), also return dict of predictions.
- `save_comparison_table(results_df)` → print it, save to `COMPARISON_CSV`.
- `plot_accuracy(results_df)` → Fig 4 bar chart with value labels, title "Figure 4: Model Accuracy Comparison".
- `plot_prf(results_df)` → Fig 5 grouped bar (Precision/Recall/F1), title "Figure 5: Precision, Recall & F1 Comparison".
- `plot_confusion(best_name, models, X_test, y_test)` → Fig 6 confusion-matrix heatmap for best model, title "Figure 6: Confusion Matrix — <best>".
- `save_best(models, results_df, scaler)` → best = highest accuracy; joblib.dump best model, scaler, and metadata `{best_model_name, feature_order, metrics}`.

---

### Task 8: `main.py`

Linear orchestration with a short header comment and step prints:
1. If `DATA_PATH` missing → call `make_sample_data()` and warn it's synthetic.
2. `df = load_data(); df = clean_and_encode(df)`.
3. `run_eda(df)`.
4. `X_train, X_test, y_train, y_test, scaler = split_and_scale(df)`.
5. `models = train_models(build_models(), X_train, y_train)`.
6. `results, _ = evaluate_models(models, X_test, y_test)`.
7. `save_comparison_table(results)`, `plot_accuracy`, `plot_prf`.
8. `best = results["Accuracy"].idxmax()`; `plot_confusion(best, ...)`; `save_best(...)`.
9. Print where artifacts were written.

---

### Task 9: `app.py` (Streamlit)

- `st.set_page_config`, title, intro.
- Guard: if artifacts missing → `st.warning("Run python main.py first")` and stop.
- Load `best_model`, `scaler`, `metadata` (cache with `@st.cache_resource`).
- Sidebar/form inputs for all 11 fields (number_inputs + selectboxes for education/self_employed).
- On submit: assemble row in `FEATURE_ORDER`, encode categoricals via config maps, `scaler.transform`, `predict` + `predict_proba`; show Approved (green) / Rejected (red) + confidence.
- "Model Performance" section: `st.dataframe(pd.read_csv(COMPARISON_CSV))`, `st.image(FIG4)`, `st.image(FIG6)`, note the best model name.

---

### Task 10: `README.md`

Project summary, dataset note (leading-space quirk + replace synthetic CSV with real
Kaggle file in `data/`), folder map, and the two run commands.

---

### Task 11: Verify

- `python main.py` → expect: prints comparison table; `report_assets/` has fig1–fig6 PNGs
  + `model_comparison.csv`; `models/` has best_model/scaler/metadata `.joblib`. No errors.
- `python -c "import ast; ast.parse(open('app.py').read())"` (syntax check) and a headless
  import smoke test of app dependencies. Full `streamlit run app.py` is user-launched.

## Self-review

- Spec coverage: all 7 user steps + all 6 PDF visuals + comparison table + best-model
  confusion matrix + joblib save + Streamlit dashboard map to tasks above. ✓
- No placeholders: each task states concrete functions, filenames, titles. ✓
- Type consistency: `FEATURE_ORDER`, encoding maps, artifact paths all come from
  `config.py` and are reused verbatim in preprocessing, evaluation, and app. ✓
