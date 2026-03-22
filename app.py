import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Wireless Channel AI", layout="wide")

st.title("📡 AI-Based Wireless Channel Prediction (Pro Version)")

# ---------------------------
# SIDEBAR SETTINGS
# ---------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "LSTM", "Compare Both"]
)

window_size = st.sidebar.slider("Window Size", 3, 15, 5)

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("data.csv")

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
# RANDOM FOREST MODEL
# ---------------------------
def run_rf(signal):
    X, y = create_dataset(signal, window_size)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    pred = model.predict(X)
    return y, pred

# ---------------------------
# LSTM MODEL
# ---------------------------
def run_lstm(signal):
    scaler = MinMaxScaler()
    signal_scaled = scaler.fit_transform(signal.reshape(-1,1))

    X, y = create_dataset(signal_scaled, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=8, verbose=0)

    pred = model.predict(X)

    pred = scaler.inverse_transform(pred)
    y = scaler.inverse_transform(y)

    return y, pred

# ---------------------------
# MODEL EXECUTION
# ---------------------------
if model_choice == "Random Forest":
    y, pred = run_rf(signal)

elif model_choice == "LSTM":
    y, pred = run_lstm(signal)

else:
    y_rf, pred_rf = run_rf(signal)
    y_lstm, pred_lstm = run_lstm(signal)

# ---------------------------
# METRICS
# ---------------------------
st.subheader("📊 Performance Metrics")

if model_choice != "Compare Both":
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)

    col1, col2 = st.columns(2)
    col1.metric("MSE", round(mse, 4))
    col2.metric("R² Score", round(r2, 4))

# ---------------------------
# PLOT
# ---------------------------
st.subheader("📈 Prediction Graph")

fig, ax = plt.subplots(figsize=(10,5))

if model_choice == "Compare Both":
    ax.plot(y_rf, label="Actual")
    ax.plot(pred_rf, label="RF Prediction")
    ax.plot(pred_lstm, label="LSTM Prediction")
else:
    ax.plot(y, label="Actual")
    ax.plot(pred, label="Predicted")

ax.legend()
ax.set_title("Wireless Channel Prediction")

st.pyplot(fig)

# ---------------------------
# DOWNLOAD BUTTON
# ---------------------------
st.subheader("⬇️ Download Predictions")

if model_choice != "Compare Both":
    result_df = pd.DataFrame({
        "Actual": y.flatten(),
        "Predicted": pred.flatten()
    })
else:
    result_df = pd.DataFrame({
        "Actual": y_rf.flatten(),
        "RF_Predicted": pred_rf.flatten(),
        "LSTM_Predicted": pred_lstm.flatten()
    })

csv = result_df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="predictions.csv",
    mime="text/csv"
)