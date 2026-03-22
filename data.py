import numpy as np
import matplotlib.pyplot as plt

def generate_signal():
    time = np.arange(0, 100, 0.1)
    signal = np.sin(time) + np.random.normal(0, 0.2, len(time))
    return time, signal

# optional: run this file alone to see raw signal
if __name__ == "__main__":
    time, signal = generate_signal()

    plt.plot(time, signal)
    plt.title("Simulated Wireless Channel")
    plt.xlabel("Time")
    plt.ylabel("Signal Strength")
    plt.show()