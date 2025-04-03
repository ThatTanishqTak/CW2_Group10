
# Stock Price Prediction using GRU (Gated Recurrent Unit)

This project demonstrates how to build a time series forecasting model using GRU (Gated Recurrent Unit) for predicting stock prices. The example uses Starbucks stock data, but the steps can be easily adapted for other stocks or models like LSTM, CNN, or Transformer-based approaches.

## Project Overview

**Goal**: Predict future stock prices based on historical data.

**Model**: GRU (Gated Recurrent Unit), a type of Recurrent Neural Network (RNN) designed for sequential data such as time series.

**Dataset**: `starbucks_stock.csv` containing the columns `Date` and `Open` price (you can modify the code for other columns or datasets).

---

## Steps to Reproduce

### 1. Setup Environment

Ensure you have the required libraries installed by running:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

### 2. Data Preprocessing

First, load the dataset and handle any missing values (if any). Normalize the data to scale values between 0 and 1 using MinMaxScaler. Then, create sequences for training the model.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('starbucks_stock.csv', parse_dates=['Date'], index_col='Date')
stock_prices = data[['Open']].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_prices)

# Create sequences (X: input, y: output)
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, time_step=60)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for GRU (samples, timesteps, features)
```

### 3. Split Data into Train/Test Sets

Split the data into training and testing sets. Use the `train_test_split` method from `sklearn`:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```

### 4. Build the GRU Model

Define the GRU-based model using Keras. This model includes two GRU layers with dropout for regularization:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')
```

### 5. Train the Model

Train the model using the training data and validate it with the testing data:

```python
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```

### 6. Evaluate and Visualize Results

After training the model, evaluate its performance by predicting the stock prices on the test set and plotting the results:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Predictions
y_pred = model.predict(X_test)
y_pred_scaled = scaler.inverse_transform(y_pred)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_test_scaled, color='blue', label='Actual Price')
plt.plot(y_pred_scaled, color='red', label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Metrics
mse = mean_squared_error(y_test_scaled, y_pred_scaled)
mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
```

---

## Adapting to Other Models

To use a different model (e.g., LSTM, CNN, or Transformer), you can modify the model architecture by replacing the GRU layers with your preferred model type. You can also adjust hyperparameters (e.g., number of units, dropout rate) as needed.

### Example for LSTM:

```python
from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
```

---

## Dataset

Place the dataset (`starbucks_stock.csv`) in the project directory. The dataset should have the following columns:

- **Date**: Date of the stock data
- **Open**: Opening price of Starbucks stock for the given day

You can adjust the code to use different stock data or columns as needed.

---

## License

This project is open-source and available under the MIT License. Feel free to contribute and modify it according to your needs.
