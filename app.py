import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("sonar_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(
    page_title="Sonar Rock vs Mine Prediction",
    page_icon="🛥️",
    layout="centered"
)

st.title("🛥️ Sonar Rock vs Mine Prediction")
st.write(
    "Upload a CSV file containing **60 sonar signal features** "
    "to predict whether the object is a **Rock** or a **Mine**."
)

st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "Upload Sonar CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file, header=None)

        st.write("### Uploaded Data Preview")
        st.dataframe(input_df.head())

        # Validate input shape
        if input_df.shape[1] != 60:
            st.error("❌ CSV file must contain exactly 60 feature columns.")
        else:
            # Scale input
            X_scaled = scaler.transform(input_df.values)

            # Predict
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]

            # Convert predictions
            prediction_labels = [
                "Mine" if p == 1 else "Rock" for p in predictions
            ]

            # Output dataframe
            output_df = input_df.copy()
            output_df["Prediction"] = prediction_labels
            output_df["Mine_Probability"] = probabilities

            st.markdown("### Prediction Results")
            st.dataframe(output_df.head())

            # Download button
            csv = output_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv,
                file_name="sonar_predictions.csv",
                mime="text/csv"
            )

            st.success("✅ Prediction completed successfully!")

    except Exception as e:
        st.error(f"Error: {e}")
