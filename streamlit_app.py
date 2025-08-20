import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# -----------------------------
# Load model & scaler directly
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")   # path in repo

model = load_model()

try:
    scaler = joblib.load("scaler.pkl")  # if you have a scaler.pkl
    st.sidebar.success("Scaler loaded from scaler.pkl")
except:
    scaler = None
    st.sidebar.warning("No scaler.pkl found – will fit one on inputs.")

# -----------------------------
# User interface
# -----------------------------
st.title("🌧️ Rainfall Forecast App")

st.write("Enter rainfall (mm) for the last 3 days to get forecast:")

d1 = st.number_input("Day 1 rainfall (mm)", min_value=0.0, step=0.1)
d2 = st.number_input("Day 2 rainfall (mm)", min_value=0.0, step=0.1)
d3 = st.number_input("Day 3 rainfall (mm)", min_value=0.0, step=0.1)

if st.button("Forecast"):
    X_input = np.array([[d1, d2, d3]])

    # If model is LSTM, reshape as (1, timesteps, features)
    if len(model.input_shape) == 3:
        X_input = X_input.reshape((1, 3, 1))

    # Scale if scaler is available
    if scaler:
        X_scaled = scaler.transform(np.log1p(X_input))
    else:
        # quick-fit scaler if not available
        X_scaled = np.log1p(X_input)

    pred = model.predict(X_input)  # or model.predict(X_scaled) if training used scaling
    st.success(f"Forecast: {pred[0][0]:.2f} mm")
