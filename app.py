
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "best_fraud_detection_pipeline.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Machine Learning powered transaction analysis")

st.divider()

st.subheader("Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(data.head())

        missing = [
            col for col in FEATURES
            if col not in data.columns
        ]

        if missing:
            st.error(f"Missing columns: {missing}")

        elif data.empty:
            st.warning("The uploaded file is empty.")

        else:
            X = data[FEATURES].apply(
                pd.to_numeric,
                errors="coerce"
            )

            if X.isnull().any().any():
                st.error(
                    "Dataset contains missing or non-numeric values."
                )

            else:
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]

                results = data.copy()
                results["Prediction"] = predictions
                results["Fraud_Probability"] = probabilities
                results["Status"] = [
                    "Fraud" if p == 1 else "Normal"
                    for p in predictions
                ]

                total = len(results)
                fraud = int((predictions == 1).sum())
                normal = total - fraud

                st.divider()
                st.subheader("Prediction Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Transactions", total)
                col2.metric("Normal Transactions", normal)
                col3.metric("Potential Fraud", fraud)

                st.subheader("Prediction Results")
                st.dataframe(results, use_container_width=True)

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Results",
                    csv,
                    "fraud_detection_results.csv",
                    "text/csv"
                )

                st.success("Analysis completed successfully!")

    except Exception as e:
        st.error(f"Error processing CSV: {e}")

else:
    st.info("Upload a transaction CSV file to begin.")
