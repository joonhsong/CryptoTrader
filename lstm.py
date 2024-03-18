'''

This code predicts the future value of a crypto currency using the Long Short Term Memory (LSTM)
model. The model is trained from the past 1 week data of the coin price every hour. 

© Written by Joon Song

'''

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pyupbit
import time
import datetime

# Returns the current balance
def get_balance(ticker):
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

# Returns the start time of the market
def get_start_time(ticker):
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0]
    return start_time

# Returns the current price of ticker
def get_current_price(ticker):
    return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"]

# Returns the predicted value of coin after 1 hour by lstm prediction based on
# past week's data
def lstm_prediction():
    # Define constants
    LOOKBACK_SIZE = 160
    PREDICT_FUTURE = 1

    # Load historical data from Upbit API
    df = pyupbit.get_ohlcv("KRW-BCH", interval="minute60", count=168)

    # Get the current price
    current_price = pyupbit.get_current_price("KRW-BCH")

    # Create a new DataFrame for the current price
    df_current = pd.DataFrame({'close': [current_price]}, index=[pd.to_datetime('now')])

    # Concatenate the current price with the historical data
    df = pd.concat([df, df_current])

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for i in range(LOOKBACK_SIZE, len(scaled_data) - PREDICT_FUTURE + 1):
        x_train.append(scaled_data[i - LOOKBACK_SIZE:i, 0])
        y_train.append(scaled_data[i + PREDICT_FUTURE - 1:i + PREDICT_FUTURE, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Use the model to predict the next price
    last_days_data = scaled_data[-LOOKBACK_SIZE:]
    next_price = model.predict(last_days_data.reshape(1, -1, 1))
    next_price = scaler.inverse_transform(next_price)

    return next_price

access = "key"
secret = "keyy"

upbit = pyupbit.Upbit(access, secret)
print("autotrade start")

# Start Auto trade
while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-BCH")
        end_time = start_time + datetime.timedelta(days=1)
        bought = 0

        if start_time < now < end_time - datetime.timedelta(seconds=10):
            pred_price = lstm_prediction()
            current_price = get_current_price("KRW-BCH")
            if pred_price > current_price:
                krw = get_balance("KRW")
                if krw > 5000:
                    upbit.buy_market_order("KRW-BCH", krw*0.9995)
                    bought = 1
            elif pred_price == current_price and bought == 1:
                coin = get_balance("BCH")
                if coin > 0:
                    upbit.sell_market_order("KRW-BCH", coin*0.9995)
                    bought = 0
        else:
            coin = get_balance("BCH")
            if coin > 0:
                upbit.sell_market_order("KRW-BCH", coin*0.9995)
                bought = 0
            break

    except Exception as e:
        print(e)
        time.sleep(1)

