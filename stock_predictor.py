# EBPL-DS – Cracking the Market Code with AI-Driven Stock Price Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Download stock data
def load_data(ticker='AAPL', start='2015-01-01', end='2024-12-31'):
    df = yf.download(ticker, start=start, end=end)
    return df[['Close']]

# Step 2: Preprocess the data
def preprocess(data, time_step=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_step, len(scaled)):
        X.append(scaled[i-time_step:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

# Step 3: Build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Plot predictions
def plot_results(real, predicted, ticker):
    plt.figure(figsize=(10,6))
    plt.plot(real, label="Actual Price")
    plt.plot(predicted, label="Predicted Price")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Step 5: Main function
def main():
    ticker = input("Enter stock symbol (e.g., AAPL, TSLA, MSFT): ").upper() or "AAPL"
    raw_data = load_data(ticker)
    X, y, scaler = preprocess(raw_data.values)
    
    model = build_model((X.shape[1], 1))
    print("Training the model...")
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    
    predictions = model.predict(X)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y.reshape(-1,1))

    plot_results(actual_prices, predicted_prices, ticker)

if __name__ == "__main__":
    main()
