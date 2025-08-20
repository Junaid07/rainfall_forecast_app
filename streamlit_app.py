import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os

# -----------------------------
# Load model & scaler
# -----------------------------
@st.cache_resource
def load_model():
    if os.path.exists("model.keras"):
        return tf.keras.models.load_model("model.keras", compile=False)
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

# Try loading scaler
scaler = None
if os.path.exists("scaler.pkl"):
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception:
        scaler = None

# -----------------------------
# User interface
# -----------------------------
st.title("🌧️ Rainfall Forecast (Next Day)")
st.markdown("Enter rainfall (mm) for the **last 3 days** to forecast tomorrow’s rainfall.")

d1 = st.number_input("Day 1 (oldest)", min_value=0.0, step=0.1)
d2 = st.number_input("Day 2", min_value=0.0, step=0.1)
d3 = st.number_input("Day 3 (most recent)", min_value=0.0, step=0.1)

if st.button("Forecast"):
    X_input = np.array([[d1, d2, d3]])

    # Reshape for LSTM input
    if len(model.input_shape) == 3:
        X_input = X_input.reshape((1, 3, 1))

    # Apply scaler if available
    if scaler is not None:
        X_scaled = scaler.transform(np.log1p(X_input))
        pred_scaled = model.predict(X_scaled, verbose=0)
        forecast = np.expm1(scaler.inverse_transform(pred_scaled))[0, 0]
    else:
        # No scaler, run raw inputs
        pred = model.predict(X_input, verbose=0)
        forecast = float(pred[0, 0])

    st.success(f"🌤️ Forecast for next day: **{forecast:.2f} mm**")
    st.caption("This is an AI-based rainfall forecast using your last 3 days of data.")
