import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------------------
# Step 1: Load Dataset
# ---------------------------
data = pd.read_csv("data.csv")
signal = data["signal"].values.reshape(-1, 1)

# ---------------------------
# Step 2: Normalize Data
# ---------------------------
scaler = MinMaxScaler()
signal_scaled = scaler.fit_transform(signal)

# ---------------------------
# Step 3: Create Dataset
# ---------------------------
def create_dataset(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X, y = create_dataset(signal_scaled, 5)

# reshape for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# ---------------------------
# Step 4: Train-Test Split
# ---------------------------
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------------
# Step 5: Build LSTM Model
# ---------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# ---------------------------
# Step 6: Train Model
# ---------------------------
model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=1)

# ---------------------------
# Step 7: Predictions
# ---------------------------
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# inverse scaling
train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)

y_train_actual = scaler.inverse_transform(y_train)
y_test_actual = scaler.inverse_transform(y_test)

# ---------------------------
# Step 8: Evaluation
# ---------------------------
mse = mean_squared_error(y_test_actual, test_pred)
r2 = r2_score(y_test_actual, test_pred)

print("LSTM Test MSE:", mse)
print("LSTM R2 Score:", r2)

# ---------------------------
# Step 9: Plot
# ---------------------------
plt.figure(figsize=(10,5))

plt.plot(y_test_actual, label="Actual")
plt.plot(test_pred, label="Predicted")

plt.title("LSTM Wireless Channel Prediction")
plt.xlabel("Time Step")
plt.ylabel("Signal")

plt.legend()
plt.show()