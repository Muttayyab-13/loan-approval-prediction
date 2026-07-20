"""
config.py — one place for all settings used across the project.

Both the training pipeline (main.py) and the dashboard (app.py) import from here,
so the app always encodes user input exactly the same way the model was trained.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders and file paths
# ---------------------------------------------------------------------------
# BASE_DIR is the project root (the folder that contains this "src" folder).
# Building paths from BASE_DIR means the scripts work no matter where you run them.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"            # raw dataset lives here
MODELS_DIR = BASE_DIR / "models"        # saved model + scaler go here
ASSETS_DIR = BASE_DIR / "report_assets"  # all charts + comparison table go here

# Make sure the output folders exist (does nothing if they already do).
for folder in (DATA_DIR, MODELS_DIR, ASSETS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "loan_approval_dataset.csv"

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
TARGET = "loan_status"            # what we predict (Approved / Rejected)
DROP_COLS = ["loan_id"]           # just an ID, not useful for prediction

# The 9 columns that are already numbers.
NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

# The 2 text columns we turn into numbers.
CATEGORICAL_FEATURES = ["education", "self_employed"]

# Exact order of features the model is trained on.
# app.py must build its input row in THIS same order.
FEATURE_ORDER = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

# ---------------------------------------------------------------------------
# Encoding maps (text -> number)
# ---------------------------------------------------------------------------
EDUCATION_MAP = {"Graduate": 1, "Not Graduate": 0}
SELF_EMPLOYED_MAP = {"Yes": 1, "No": 0}
TARGET_MAP = {"Approved": 1, "Rejected": 0}

# ---------------------------------------------------------------------------
# Saved-artifact paths
# ---------------------------------------------------------------------------
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.joblib"
COMPARISON_CSV = ASSETS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Figure file paths (PNG charts saved for the report and dashboard)
# ---------------------------------------------------------------------------
FIG1_TARGET = ASSETS_DIR / "fig1_target_distribution.png"
FIG2_FEATURE = ASSETS_DIR / "fig2_feature_distribution.png"
FIG3_HEATMAP = ASSETS_DIR / "fig3_correlation_heatmap.png"
FIG4_ACCURACY = ASSETS_DIR / "fig4_accuracy_comparison.png"
FIG5_PRF = ASSETS_DIR / "fig5_precision_recall_f1.png"
FIG6_CONFUSION = ASSETS_DIR / "fig6_confusion_matrix.png"
