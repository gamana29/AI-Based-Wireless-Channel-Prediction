import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Generate sine wave data
t = np.linspace(0, 100, 1000)
data = np.sin(t)

# 2. Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# 3. Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 20
X, y = create_sequences(data_scaled, sequence_length)

# 4. Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(64),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 6. Train model
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

# 7. Predict future values
predictions = []

# Start with last test sequence
current_seq = X_test[0]

for _ in range(200):
    pred = model.predict(current_seq.reshape(1, sequence_length, 1), verbose=0)
    predictions.append(pred[0][0])
    
    # Slide window
    current_seq = np.append(current_seq[1:], pred, axis=0)

# Convert back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 8. Plot results
plt.figure(figsize=(12, 6))

# Original signal
plt.plot(t, data, label="Original Signal")

# Predicted signal (future)
future_t = np.linspace(t[-1], t[-1] + 20, 200)
plt.plot(future_t, predictions, label="Predicted Signal (LSTM)", color='orange')

plt.title("Accurate LSTM Wireless Channel Prediction")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.legend()
plt.grid()

plt.show()