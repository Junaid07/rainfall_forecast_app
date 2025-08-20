import io
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datetime import datetime, timedelta
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="3-Day Input Rainfall Forecast", layout="centered")

# -----------------------------
# Helpers
# -----------------------------
def load_model_from_bytes(file_bytes: bytes):
    bio = io.BytesIO(file_bytes)
    return keras.models.load_model(bio)

def roll_forecast(model, window, n_steps: int):
    preds, w = [], window.copy()
    for _ in range(n_steps):
        nxt = model.predict(w, verbose=0)        # (1,1)
        preds.append(nxt[0,0])
        w = np.concatenate([w[:,1:,:], nxt.reshape(1,1,1)], axis=1)
    return np.array(preds).reshape(-1, 1)

def invert_to_mm(pred_scaled: np.ndarray, scaler: MinMaxScaler):
    inv = scaler.inverse_transform(pred_scaled)  # back to log1p(mm)
    return np.expm1(inv)                         # to mm

def plot_forecast(last_vals, fc_df):
    last_idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(last_vals), freq="D")
    df_hist = pd.DataFrame({"rain_mm": last_vals}, index=last_idx)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist["rain_mm"],
                             name="Last days (mm)", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df["forecast_mm"],
                             name="Forecast (mm)", mode="lines+markers"))
    fig.update_layout(template="plotly_white",
                      yaxis_title="mm/day",
                      xaxis_title="Date",
                      margin=dict(l=10,r=10,t=40,b=10),
                      legend=dict(orientation="h", y=1.12))
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("🌧️ Rainfall Forecast from Last 3 Days")

st.markdown(
    "Upload your **Keras `.h5` LSTM** model, enter the **last 3 days** of rainfall (mm), "
    "and get a multi-day forecast.\n\n"
    "If you trained with a scaler, upload your `scaler.pkl`. Otherwise, the app will fit "
    "MinMax on `log1p(rain)` using the values you provide."
)

colA, colB = st.columns(2)
with colA:
    uploaded_model = st.file_uploader("Upload model (.h5)", type=["h5"])
with colB:
    uploaded_scaler = st.file_uploader("Upload scaler (joblib .pkl) [optional]", type=["pkl"])

st.divider()

# Model/training lookback
col1, col2 = st.columns(2)
with col1:
    model_lookback = st.number_input("Model lookback (training)", min_value=1, max_value=120, value=3, step=1)
with col2:
    n_future = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=7, step=1)

# Inputs: 3 days by default
st.subheader("Enter last 3 days of rainfall (mm)")
use_more = st.checkbox("Need to enter more than 3 days? (e.g., if model_lookback > 3)")

if use_more:
    num_days = model_lookback
    st.caption(f"Enter exactly {num_days} values (oldest → newest).")
else:
    num_days = 3
    if model_lookback != 3:
        st.info(f"Your model lookback is {model_lookback}. "
                f"We'll **pad**/truncate to match, but it’s best to enter exactly {model_lookback} values.")

cols = st.columns(min(6, num_days))
vals = []
for i in range(num_days):
    with cols[i % len(cols)]:
        v = st.number_input(f"Day {i+1}", min_value=0.0, value=0.0, step=0.1, format="%.1f")
        vals.append(v)

# Action
if st.button("Forecast"):
    # Checks
    if uploaded_model is None:
        st.error("Please upload your `.h5` model.")
        st.stop()

    try:
        model = load_model_from_bytes(uploaded_model.read())
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Prepare the latest input window to match model_lookback
    last_vals = np.array(vals, dtype=float)
    if len(last_vals) < model_lookback:
        # pad at the front with zeros if user entered fewer values than the model expects
        pad = np.zeros(model_lookback - len(last_vals), dtype=float)
        last_vals = np.concatenate([pad, last_vals])
    elif len(last_vals) > model_lookback:
        # keep only the most recent 'model_lookback' values
        last_vals = last_vals[-model_lookback:]

    # Build a tiny "history" to fit or apply scaler in log1p space
    series_log1p = np.log1p(last_vals).reshape(-1, 1)

    # Load/fit scaler
    if uploaded_scaler is not None:
        try:
            scaler = joblib.load(uploaded_scaler)
            st.success("Loaded scaler from .pkl")
        except Exception as e:
            st.error(f"Failed to load scaler: {e}")
            st.stop()
    else:
        scaler = MinMaxScaler().fit(series_log1p)
        st.warning("No scaler uploaded. Fitted MinMax on the provided values (log1p space).")

    # Transform to scaled space and build window tensor
    scaled = scaler.transform(series_log1p)              # shape (lookback, 1)
    window = scaled.reshape(1, model_lookback, 1)       # (1, lookback, 1)

    # Forecast
    try:
        pred_scaled = roll_forecast(model, window, n_steps=int(n_future))
    except Exception as e:
        st.error(f"Model prediction failed. Likely input shape mismatch. Details: {e}")
        st.stop()

    pred_mm = invert_to_mm(pred_scaled, scaler).ravel()

    # Build forecast index starting “tomorrow”
    start_future = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    idx_future = pd.date_range(start=start_future, periods=int(n_future), freq="D")
    forecast_df = pd.DataFrame({"forecast_mm": pred_mm}, index=idx_future)

    # Plot
    fig = plot_forecast(last_vals, forecast_df)
    st.plotly_chart(fig, use_container_width=True)

    # Table + download
    st.dataframe(forecast_df.style.format({"forecast_mm": "{:.2f}"}))
    st.download_button(
        "Download forecast CSV",
        data=forecast_df.to_csv().encode("utf-8"),
        file_name="rain_forecast.csv",
        mime="text/csv",
    )

    # Small notes
    st.caption(
        "Notes: Inputs are assumed to be daily rainfall (mm). "
        "Preprocessing is `log1p` → MinMax; upload your scaler if you used a different one in training. "
        "The model input tensor is shaped as (1, lookback, 1)."
    )
