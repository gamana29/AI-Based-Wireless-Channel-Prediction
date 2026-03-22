import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------------------
# LOAD DATA
# ---------------------------
data = pd.read_csv("data.csv")

signal = data["signal"].values.reshape(-1, 1)

# ---------------------------
# NORMALIZATION
# ---------------------------
scaler = MinMaxScaler()
signal_scaled = scaler.fit_transform(signal)

# ---------------------------
# CREATE DATASET
# ---------------------------
def create_dataset(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 5
X, y = create_dataset(signal_scaled, window_size)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# ---------------------------
# BUILD MODEL
# ---------------------------
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# ---------------------------
# TRAIN MODEL
# ---------------------------
print("Training LSTM model...")
model.fit(X, y, epochs=15, batch_size=8, verbose=1)

# ---------------------------
# PREDICTIONS
# ---------------------------
predictions = model.predict(X)

# Convert back to original scale
predictions = scaler.inverse_transform(predictions)
y_actual = scaler.inverse_transform(y)

# ---------------------------
# METRICS
# ---------------------------
mse = mean_squared_error(y_actual, predictions)
r2 = r2_score(y_actual, predictions)

print("\n📊 LSTM Results:")
print("MSE:", mse)
print("R² Score:", r2)

# ---------------------------
# PLOT
# ---------------------------
plt.figure(figsize=(10,5))
plt.plot(y_actual, label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("LSTM Wireless Channel Prediction")
plt.legend()
plt.show()