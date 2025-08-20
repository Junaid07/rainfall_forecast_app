import os
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(page_title="🌧️ Rainfall Forecast App", layout="centered")

# -----------------------------
# Load model (prefer .keras; fall back to .h5) — do NOT compile
# -----------------------------
@st.cache_resource
def load_model():
    if os.path.exists("model.keras"):
        return tf.keras.models.load_model("model.keras", compile=False)
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

# -----------------------------
# Load scaler if present (optional)
# -----------------------------
scaler = None
try:
    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
        st.sidebar.success("Scaler loaded from scaler.pkl")
    else:
        st.sidebar.warning("No scaler.pkl found – will fit/assume preprocessing from inputs.")
except Exception as e:
    st.sidebar.error(f"Failed to load scaler.pkl: {e}")

# -----------------------------
# UI
# -----------------------------
st.title("🌧️ Rainfall Forecast (Last 3 Days → Next Day)")

st.write("Enter rainfall (mm) for the **last 3 days** to forecast the **next day**.")

c1, c2, c3 = st.columns(3)
with c1:
    d1 = st.number_input("Day 1 (oldest)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
with c2:
    d2 = st.number_input("Day 2", min_value=0.0, step=0.1, value=0.0, format="%.1f")
with c3:
    d3 = st.number_input("Day 3 (most recent)", min_value=0.0, step=0.1, value=0.0, format="%.1f")

st.caption("Note: If your model was trained on **log1p → MinMax** scaled inputs, keep a matching `scaler.pkl` in the repo. "
           "Without it, this app will either pass raw mm or quick-fit a scaler on the 3 values depending on your choice below.")

use_scaling = st.checkbox("Apply log1p + scaler (recommended if used in training)", value=bool(scaler is not None))

# -----------------------------
# Inference
# -----------------------------
if st.button("Forecast"):
    # Assemble feature vector
    X = np.array([[d1, d2, d3]], dtype="float32")  # shape (1,3)

    # If the model expects 3D (LSTM style), reshape to (1, timesteps, features)
    if len(model.input_shape) == 3:
        X = X.reshape((1, 3, 1))  # (batch=1, timesteps=3, features=1)

    # Preprocessing path
    if use_scaling:
        # If a scaler is present, assume training pipeline was: log1p(mm) -> MinMax
        if scaler is not None:
            # Apply per-timestep log1p then MinMax using the training scaler
            X_scaled = np.log1p(X)
            # scaler expects 2D; reshape, transform, back to model shape
            if X_scaled.ndim == 3:
                b, t, f = X_scaled.shape
                X_scaled2d = X_scaled.reshape(-1, 1)
                X_scaled2d = scaler.transform(X_scaled2d)
                X_in = X_scaled2d.reshape(b, t, f)
            else:
                X_in = scaler.transform(np.log1p(X))  # 2D case
        else:
            # No scaler.pkl → quick-fit a temporary scaler on the 3 values (least ideal but works)
            from sklearn.preprocessing import MinMaxScaler
            if X.ndim == 3:
                tmp2d = np.log1p(X).reshape(-1, 1)
                tmp_scaler = MinMaxScaler().fit(tmp2d)
                X_in = tmp_scaler.transform(tmp2d).reshape(X.shape)
            else:
                tmp_scaler = MinMaxScaler().fit(np.log1p(X))
                X_in = tmp_scaler.transform(np.log1p(X))
    else:
        # No scaling: pass raw mm directly
        X_in = X

    # Predict
    try:
        yhat = model.predict(X_in, verbose=0)
    except Exception as e:
        st.error(f"Prediction failed. Likely input-shape or preprocessing mismatch.\nDetails: {e}")
        st.stop()

    # If the model outputs scaled/log space, you may need to invert here.
    # Since we don't know your exact training pipeline, we show raw model output:
    pred_val = float(yhat.reshape(-1)[0])

    st.success(f"Forecast for next day: **{pred_val:.2f}** (model output units)")

    st.caption(
        "If your model predicts *scaled/log* values, add the correct inverse transform here "
        "(e.g., `expm1` and/or `scaler.inverse_transform`). "
        "For a fully consistent pipeline, consider baking preprocessing into the model before saving."
    )
