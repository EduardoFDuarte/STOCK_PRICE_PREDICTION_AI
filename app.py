import streamlit as st

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras.backend as K
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error

# Set random seed for reproducibility
np.random.seed(10)

# Function to create dataset
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1

# Streamlit app
st.title('Stock Price Prediction')

# Sidebar for user input
st.sidebar.header('Select Stock and Time Period')
stock_symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL):', 'AAPL')
time_period = st.sidebar.selectbox('Select Time Period (The best Time Period is 3 years): 3y', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '10y', 'ytd', '3y'])

# Download stock data
stock_data = yf.download(stock_symbol, period=time_period)
gold_close = stock_data['Close'].values.reshape(-1, 1)

# Scale data to range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
gold_close_scaled = scaler.fit_transform(gold_close)

# Split data into train and test sets
train_size = math.ceil(len(gold_close_scaled) * 0.85)
train_data, test_data = gold_close_scaled[0:train_size], gold_close_scaled[train_size:]

# Define look back
look_back = 1

# Create dataset for LSTM
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)


# Define LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the model
model.fit(trainX, trainY, epochs=100, batch_size=16, verbose=2)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions to original scale
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

# Plot the results
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Historical Data'))

fig.add_trace(
    go.Scatter(x=stock_data.index[:len(trainPredict)], y=trainPredict.flatten(), mode='lines', name='Predicted Prices for the Train Set'))

fig.add_trace(
    go.Scatter(x=stock_data.index[len(trainPredict)+1:len(trainPredict)+1+len(testPredict)], y=testPredict.flatten(), mode='lines', name='Predicted Prices for the Test Set'))

fig.update_layout(
    title_text="Historical vs Predicted Prices",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

st.plotly_chart(fig)

# Calculate percentage error
trainMape = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
testMape = mean_absolute_percentage_error(testY[0], testPredict[:,0])

# Display scores
st.write('Train RMSE:', trainScore)
st.write('Test RMSE:', testScore)
st.write('Train Percentage Error:', trainMape * 100)
st.write('Test Percentage Error:', testMape * 100)





import pandas as pd


# Define a function to determine if the price is going up or down
def predict_price_direction(predictions, actual_prices):
    directions = []
    for pred, prev_price in zip(predictions, actual_prices):
        if pred > 0:
            directions.append("Up")
        elif pred < 0:
            directions.append("Down")
        else:
            directions.append("No Change")
    return directions

# Streamlit app
st.title('Price Direction Prediction')

# Define a function to determine if the price is going up or down
def predict_price_direction(predictions, actual_prices):
    directions = []
    for pred, prev_price in zip(predictions, actual_prices):
        if pred > 0:
            directions.append("Up")
        elif pred < 0:
            directions.append("Down")
        else:
            directions.append("No Change")
    return directions

# Combine trainPredict and testPredict into one list
allPredict = np.concatenate((trainPredict, testPredict))

# Predictions of the next week for both training and test sets
nextWeekPredict = allPredict[7]

# Calculate the differences between consecutive predictions
priceDiff = nextWeekPredict - allPredict[:-7]

# Predict whether the price is going up or down
priceDirections = predict_price_direction(priceDiff, allPredict[:-7])

# Calculate the differences between consecutive prices for the last week
price_diff_last_week = gold_close[-7:] - gold_close[-14:-7]

# Predict whether the price is going up or down for each day of the last week
price_directions_last_week = predict_price_direction(price_diff_last_week, gold_close[-7:])


print("Price Directions for Each Day of Last Week:")
print(price_directions_last_week)

# Determine the overall direction of the last week
week_direction_last = max(set(price_directions_last_week), key=price_directions_last_week.count)

print("Overall Direction of Last Week:", week_direction_last)

print("Price Directions for Each Day of Last Week:")
for i, direction in enumerate(price_directions_last_week):
    print("Day", i + 1, ":", direction)

# Extracting predictions for the next week
num_days = 7
next_week_predictions = priceDirections[:num_days]

print("Predicted Price Directions for Each Day of Next Week:")
print(next_week_predictions)

# Determine the overall direction of the week
week_direction = max(set(next_week_predictions), key=next_week_predictions.count)

print("Overall Direction of Next Week:", week_direction)



print("Price Directions for Each Day of Next Week:")
for i, direction in enumerate(next_week_predictions):
    print("Day", i + 1, ":", direction)

# Output tables
st.write("Price Directions for Each Day of Last Week:")
df_last_week = pd.DataFrame({"Day": range(1, 8), "Price Direction": price_directions_last_week})
st.table(df_last_week)

st.write("Overall Direction of Last Week:", max(set(price_directions_last_week), key=price_directions_last_week.count))

st.write("Predicted Price Directions for Each Day of Next Week:")
df_next_week = pd.DataFrame({"Day": range(1, 8), "Predicted Price Direction": next_week_predictions})
st.table(df_next_week)

st.write("Overall Direction of Next Week:", max(set(next_week_predictions), key=next_week_predictions.count))
