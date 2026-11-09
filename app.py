
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD TRAINED MACHINE LEARNING MODEL
# --------------------------------------------------

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


# --------------------------------------------------
# 3. REQUIRED DATASET FEATURES
# --------------------------------------------------

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# --------------------------------------------------
# 4. APPLICATION HEADER
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "Machine Learning powered transaction analysis"
)

st.divider()


# --------------------------------------------------
# 5. UPLOAD TRANSACTION DATASET
# --------------------------------------------------

st.subheader("Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        # Read uploaded CSV
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.dataframe(
            data.head(),
            use_container_width=True
        )

        # Check required columns
        missing = [
            col for col in FEATURES
            if col not in data.columns
        ]

        if missing:

            st.error(
                f"Missing columns: {missing}"
            )

        elif data.empty:

            st.warning(
                "The uploaded file is empty."
            )

        else:

            # Convert features into numeric values
            X = data[FEATURES].apply(
                pd.to_numeric,
                errors="coerce"
            )

            # Check missing or invalid values
            if X.isnull().any().any():

                st.error(
                    "Dataset contains missing or "
                    "non-numeric values."
                )

            else:

                # ----------------------------------
                # 6. MACHINE LEARNING PREDICTIONS
                # ----------------------------------

                predictions = model.predict(X)

                probabilities = (
                    model.predict_proba(X)[:, 1]
                )

                # ----------------------------------
                # 7. CREATE PREDICTION RESULTS
                # ----------------------------------

                results = data.copy()

                results["Prediction"] = predictions

                results["Fraud_Probability (%)"] = (
                    probabilities * 100
                ).round(2)

                results["Status"] = [
                    "Potential Fraud" if p == 1
                    else "Normal"
                    for p in predictions
                ]

                # ----------------------------------
                # 8. PREDICTION SUMMARY
                # ----------------------------------

                total = len(results)

                fraud = int(
                    (predictions == 1).sum()
                )

                normal = total - fraud

                st.divider()

                st.subheader(
                    "Prediction Summary"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Transactions",
                    total
                )

                col2.metric(
                    "Normal Transactions",
                    normal
                )

                col3.metric(
                    "Potential Fraud",
                    fraud
                )

                # ----------------------------------
                # 9. PREDICTION RESULTS TABLE
                # ----------------------------------

                st.divider()

                st.subheader(
                    "Prediction Results"
                )

                # Color-code transaction status
                def highlight_status(row):

                    if row["Status"] == "Potential Fraud":

                        return [
                            "background-color: #5c2020; color: white"
                        ] * len(row)

                    return [
                        "background-color: #164d35; color: white"
                    ] * len(row)

                styled_results = (
                    results.style
                    .format({
                        "Fraud_Probability (%)": "{:.2f}%"
                    })
                    .apply(
                        highlight_status,
                        axis=1
                    )
                )

                st.dataframe(
                    styled_results,
                    use_container_width=True
                )

                # ----------------------------------
                # 10. DOWNLOAD PREDICTION RESULTS
                # ----------------------------------

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

                # ----------------------------------
                # 11. COMPLETION MESSAGE
                # ----------------------------------

                st.success(
                    "Analysis completed successfully!"
                )

                st.info(
                    "Fraud probability is the model's "
                    "estimated probability for each transaction. "
                    "It is not a guarantee of fraud."
                )

    except Exception as e:

        st.error(
            f"Error processing CSV: {e}"
        )

else:

    st.info(
        "Upload a transaction CSV file to begin."
    )
