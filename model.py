import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------
# Step 1: Generate Data
# ---------------------------
time = np.arange(0, 100, 0.1)
signal = np.sin(time) + np.random.normal(0, 0.2, len(time))

# ---------------------------
# Step 2: Create Dataset
# ---------------------------
def create_dataset(data, window_size=10):
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])

    return np.array(X), np.array(y)

X, y = create_dataset(signal, 10)

# ---------------------------
# Step 3: Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ---------------------------
# Step 4: Train Model (Better Model)
# ---------------------------
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# ---------------------------
# Step 5: Predictions
# ---------------------------
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# ---------------------------
# Step 6: Evaluation
# ---------------------------
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

r2 = r2_score(y_test, test_pred)

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("R2 Score:", r2)

# ---------------------------
# Step 7: Save Model
# ---------------------------
joblib.dump(model, "channel_model.pkl")
print("Model saved as channel_model.pkl")

# ---------------------------
# Step 8: Visualization
# ---------------------------
plt.figure(figsize=(10,5))

plt.plot(y_test[:200], label="Actual")
plt.plot(test_pred[:200], label="Predicted")

plt.title("AI Wireless Channel Prediction (Advanced)")
plt.xlabel("Time Step")
plt.ylabel("Signal")

plt.legend()
plt.show()