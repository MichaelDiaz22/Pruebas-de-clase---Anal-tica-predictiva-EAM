import streamlit as st
import pandas as pd
import joblib
import os

st.title("Prediction Application")

# File uploaders
uploaded_data_file = st.file_uploader("Upload your Excel data file", type=['xlsx'])
uploaded_encoder_file = st.file_uploader("Upload the onehot_encoder.joblib file", type=['joblib'])
uploaded_scaler_file = st.file_uploader("Upload the minmax_scaler.pkl file", type=['pkl'])
uploaded_model_file = st.file_uploader("Upload the logistic_regression_best_model.joblib file", type=['joblib'])

data_df = None
onehot_encoder = None
scaler = None
best_model = None

if uploaded_data_file is not None:
    try:
        # Load the second sheet (index 1) of the Excel file
        data_df = pd.read_excel(uploaded_data_file, sheet_name=1)
        st.write("Data loaded successfully:")
        st.dataframe(data_df.head())
    except Exception as e:
        st.error(f"Error loading data file: {e}")

if uploaded_encoder_file is not None:
    try:
        onehot_encoder = joblib.load(uploaded_encoder_file)
        st.write("One-hot encoder loaded successfully.")
    except Exception as e:
        st.error(f"Error loading encoder file: {e}")

if uploaded_scaler_file is not None:
    try:
        scaler = joblib.load(uploaded_scaler_file)
        st.write("Min-Max scaler loaded successfully.")
    except Exception as e:
        st.error(f"Error loading scaler file: {e}")

if uploaded_model_file is not None:
    try:
        best_model = joblib.load(uploaded_model_file)
        st.write("Logistic Regression Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model file: {e}")

# Now, you can add the data preprocessing and prediction logic based on whether
# data_df, onehot_encoder, scaler, and best_model are not None.
# This will be done in the next steps of the plan.
