"""
app.py — a simple Streamlit dashboard for Loan Approval Prediction.

Run with:  streamlit run app.py

It shows:
  - an input form for applicant details
  - the model's prediction (Approved / Rejected) with confidence
  - a "Model Performance" section: comparison table, accuracy chart, confusion matrix

Note: run  python main.py  first so the model and charts exist.
"""

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    FEATURE_ORDER,
    EDUCATION_MAP,
    SELF_EMPLOYED_MAP,
    BEST_MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    COMPARISON_CSV,
    FIG4_ACCURACY,
    FIG6_CONFUSION,
)

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")


# ---------------------------------------------------------------------------
# Load the saved model, scaler, and metadata (cached so it loads only once).
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, scaler, metadata


st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant's details and the model will predict Approved or Rejected.")

# If the pipeline has not been run yet, the model files won't exist. Guide the user.
if not BEST_MODEL_PATH.exists():
    st.warning("Model not found. Please run  `python main.py`  first to train the model.")
    st.stop()

model, scaler, metadata = load_artifacts()

# ---------------------------------------------------------------------------
# Input form for the applicant details
# ---------------------------------------------------------------------------
st.header("Applicant Details")

with st.form("applicant_form"):
    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input("Number of dependents", 0, 10, 2)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        income_annum = st.number_input("Annual income", 0, 100_000_000, 5_000_000, step=100_000)
        loan_amount = st.number_input("Loan amount", 0, 100_000_000, 15_000_000, step=100_000)
        loan_term = st.number_input("Loan term (years)", 1, 30, 12)

    with col2:
        cibil_score = st.slider("CIBIL score", 300, 900, 650)
        residential_assets_value = st.number_input("Residential assets value", 0, 100_000_000, 5_000_000, step=100_000)
        commercial_assets_value = st.number_input("Commercial assets value", 0, 100_000_000, 3_000_000, step=100_000)
        luxury_assets_value = st.number_input("Luxury assets value", 0, 100_000_000, 8_000_000, step=100_000)
        bank_asset_value = st.number_input("Bank assets value", 0, 100_000_000, 3_000_000, step=100_000)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build one row of input in the SAME feature order used for training.
    # Encode the two text fields using the same maps from config.py.
    input_values = {
        "no_of_dependents": no_of_dependents,
        "education": EDUCATION_MAP[education],
        "self_employed": SELF_EMPLOYED_MAP[self_employed],
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }
    input_df = pd.DataFrame([input_values])[FEATURE_ORDER]

    # Scale with the SAME scaler from training, then predict.
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    confidence = proba[prediction] * 100  # probability of the predicted class

    st.header("Prediction")
    if prediction == 1:
        st.success(f"✅ Loan Approved  (confidence: {confidence:.1f}%)")
    else:
        st.error(f"❌ Loan Rejected  (confidence: {confidence:.1f}%)")

# ---------------------------------------------------------------------------
# Model Performance section
# ---------------------------------------------------------------------------
st.header("Model Performance")
st.write(f"**Best model:** {metadata['best_model_name']}")

# Comparison table
if COMPARISON_CSV.exists():
    st.subheader("Model Comparison Table")
    st.dataframe(pd.read_csv(COMPARISON_CSV).round(3))

# Accuracy chart (Figure 4)
if FIG4_ACCURACY.exists():
    st.subheader("Accuracy Comparison")
    st.image(str(FIG4_ACCURACY))

# Confusion matrix (Figure 6)
if FIG6_CONFUSION.exists():
    st.subheader("Confusion Matrix (Best Model)")
    st.image(str(FIG6_CONFUSION))
