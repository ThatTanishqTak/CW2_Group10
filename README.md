
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
---

## Dataset

Place the dataset (`starbucks_stock.csv`) in the project directory. The dataset should have the following columns:

- **Date**: Date of the stock data
- **Open**: Opening price of Starbucks stock for the given day

You can adjust the code to use different stock data or columns as needed.

---

Here’s a detailed structure for your Group Project Report based on the assignment requirements for developing deep learning-based solutions for real-time financial forecasting using GRU models. This will help guide the report writing process.

---

## **Group Project Report: Time Series Forecasting using Deep Neural Networks (GRU)**

### 1. **Aims and Objectives of the Theme Problem**

In this section, you need to outline the problem you're addressing and the objectives of your project.

**Aims:**
- To develop a deep learning-based solution for predicting stock prices (or another financial dataset) using historical data.
- To explore the use of Gated Recurrent Units (GRU) for time series forecasting in a real-world financial setting.

**Objectives:**
- **Implement an existing deep learning model** (such as GRU) for time series forecasting.
- **Design a novel deep learning model** based on the GRU architecture to improve forecasting accuracy.
- **Train and evaluate** the models using metrics such as accuracy, Mean Squared Error (MSE), and Mean Absolute Error (MAE).
- **Compare and contrast** the performance of existing and designed models in terms of prediction accuracy, generalization ability, and computational efficiency.

---

### 2. **Analysis of the Existing DNNs**

This section focuses on reviewing current models used for time series forecasting and how they apply to your dataset.

- **Overview of Existing Models**: Discuss popular models like LSTMs, GRUs, and RNNs. For instance:
  - **LSTM**: Effective for sequence prediction tasks due to its ability to capture long-term dependencies.
  - **GRU**: A variant of LSTM that simplifies the architecture by having fewer parameters but still performs well for time series forecasting.
  - **Traditional ML Models**: Briefly discuss other models like ARIMA or Random Forest, and explain why deep learning models might be more effective for this task.

- **Performance in Time Series Forecasting**: Summarize how these models have been used in financial forecasting in existing research and how they compare with each other. For example, GRUs are often preferred over vanilla RNNs due to their ability to capture long-term dependencies while being computationally more efficient than LSTMs.

- **Limitations of Existing Models**: Identify any shortcomings of the existing DNNs, such as issues with overfitting, training difficulties, or limited scalability when handling large datasets.

---

### 3. **Analysis of the Designed DNNs**

In this section, you will describe the GRU-based model that your team has designed and how it was customized for the project.

- **GRU Architecture Design**: 
  - Describe the architecture of the GRU model used for forecasting (e.g., number of layers, number of units in each layer, activation functions, dropout layers to prevent overfitting).
  - Justify why you chose GRU over LSTM (e.g., for computational efficiency or because of the specific characteristics of your dataset).

- **Hyperparameters**: List the hyperparameters used for training (e.g., learning rate, batch size, number of epochs, optimizer used).

- **Modifications for Financial Forecasting**: Explain any specific adjustments made to the general GRU architecture to tailor it for stock price prediction (e.g., using a sliding window for time series, handling seasonality, etc.).

---

### 4. **Analysis of the Training Process**

This section should cover the methodology and challenges faced during the training process.

- **Training Data Preparation**: Discuss how you prepared the training data (e.g., splitting the dataset into training and testing sets, normalizing the data using MinMaxScaler, and reshaping the data for GRU input).
  
- **Training the Model**:
  - Explain the training process, including the optimizer and loss function used (e.g., Adam optimizer, Mean Squared Error as loss function).
  - Discuss any challenges faced during training, such as overfitting, convergence issues, or the need for hyperparameter tuning.
  - Mention any techniques used to mitigate these issues, such as using dropout layers, early stopping, or adjusting the learning rate.

- **Training Duration and Computational Resources**: Provide information on the time it took to train the model, the computational resources used (e.g., CPU vs. GPU), and any scaling strategies applied if the dataset is large.

---

### 5. **Comparative Analysis and Performance Evaluation of All the DNNs Used**

This section involves evaluating the performance of the GRU model against existing models, if applicable, and comparing their results.

- **Evaluation Metrics**: 
  - Explain the metrics used to evaluate the model’s performance, such as accuracy, Mean Squared Error (MSE), Mean Absolute Error (MAE), and visual comparison using graphs (e.g., actual vs. predicted values).
  
- **Performance Results**:
  - Present the evaluation results for the existing model(s) and your GRU model. This could include a table or graph comparing the MSE, MAE, and other relevant metrics.
  - Include graphs showing how the model's predictions align with actual stock prices or financial data.
  
- **Interpretation of Results**:
  - Analyze which model performs best and why. Discuss any trade-offs between computational efficiency and prediction accuracy.
  - Consider how well the model generalizes to unseen data (i.e., whether it overfits the training data).
  
- **Conclusion and Insights**:
  - Summarize the overall findings and how your novel GRU model compares to existing solutions.
  - Suggest possible improvements or future work, such as trying more advanced models (e.g., hybrid models) or including additional features like technical indicators or macroeconomic data.

---

### **References**

Make sure to reference all sources, including any sample code or libraries used, research papers on GRU/LSTM models, and relevant tutorials. Cite your sources properly using the citation style required by your institution.

---

By following this structure, you'll be able to provide a comprehensive analysis of your machine learning models and demonstrate a clear understanding of the methods and their effectiveness in the context of your financial forecasting project.
