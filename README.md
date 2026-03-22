# AI-Based-Wireless-Channel-Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.19.0-orange)](https://streamlit.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/gamana29/ai-wireless-channel-prediction)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Overview

This project predicts wireless channel signals over time using both classical and deep learning approaches.  
It supports:

- **Linear Regression**
- **Random Forest Regression**
- **LSTM (Long Short-Term Memory) Neural Network**  

The project works with **synthetic simulated signals** as well as **real wireless datasets** (RSSI measurements).

---

## 🧠 Features

- Time-series prediction using **LSTM**
- Recursive future prediction for long-term trends
- Synthetic sine-wave signal simulation
- Supports **Linear Regression** and **Random Forest**
- Visualization of actual vs predicted signals
- Easy-to-use **Streamlit** interface for interactive predictions

---

## 🛠️ Tech Stack

- **Python 3.x**
- **NumPy** – numerical operations
- **Pandas** – data handling
- **Matplotlib** – plotting
- **Scikit-learn** – classical ML models
- **TensorFlow / Keras** – deep learning (LSTM)
- **Streamlit** – interactive web app

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/gamana29/ai-wireless-channel-prediction.git
cd ai-wireless-channel-prediction
```

2. Install dependencies:

``` bash 
pip install -r requirements.txt

```

3. Run the Streamlit app:
``` bash
streamlit run app.py
```
4. Running LSTM Standalone:
``` bash
python lstm_model.py
```

## Code Structure
```bash
.
├── app.py              
├── lstm_model.py        
├── rf_model.py         
├── utils.py             
├── data/
│   └── data.csv         
├── requirements.txt
└── README.md
```


