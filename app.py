import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Wireless Channel AI", layout="wide")
st.title("📡 AI-Based Wireless Channel Prediction")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("⚙️ Settings")
window_size = st.sidebar.slider("Window Size", 3, 15, 5)

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    if os.path.exists("data.csv"):
        data = pd.read_csv("data.csv")
    else:
        st.error("❌ No dataset found. Please upload a CSV file.")
        st.stop()

# ---------------------------
# VALIDATION
# ---------------------------
if data.empty:
    st.error("❌ Dataset is empty")
    st.stop()

if "signal" not in data.columns:
    st.error("❌ CSV must contain 'signal' column")
    st.stop()

st.subheader("📊 Dataset Preview")
st.write(data.head())

signal = data["signal"].values

# ---------------------------
# DATASET FUNCTION
# ---------------------------
def create_dataset(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# ---------------------------
# MODEL
# ---------------------------
try:
    X, y = create_dataset(signal, window_size)

    if len(X) == 0:
        st.error("❌ Not enough data for selected window size")
        st.stop()

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    predictions = model.predict(X)

    # ---------------------------
    # METRICS
    # ---------------------------
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    col1, col2 = st.columns(2)
    col1.metric("📉 MSE", round(mse, 4))
    col2.metric("📊 R² Score", round(r2, 4))

    # ---------------------------
    # PLOT
    # ---------------------------
    st.subheader("📈 Prediction Graph")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y, label="Actual")
    ax.plot(predictions, label="Predicted")
    ax.set_title("Wireless Channel Prediction")
    ax.legend()

    st.pyplot(fig, clear_figure=True)

    # ---------------------------
    # DOWNLOAD
    # ---------------------------
    st.subheader("⬇️ Download Predictions")

    result_df = pd.DataFrame({
        "Actual": y.flatten(),
        "Predicted": predictions.flatten()
    })

    csv = result_df.to_csv(index=False)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"❌ Error occurred: {e}")