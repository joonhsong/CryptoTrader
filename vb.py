'''

This code uses volatility breakout strategy to auto trade on Upbit crypto market using Upbit API.
Volatility breakout stratey uses the difference between the highest price and the lowest price
of the previous day to set the target price to buy at today. This code does this by multiplying 
k value to the difference, then adding it to the closing price of the previous day (k value is
between 0 and 1). The k value is found by analyzing the last 7 days' data to calculate which k 
would have made the most profit. This is done as it ensures that there is an upward momentum
when the price has risen from the closing price of yesterday by k*difference.

© Written by Joon Song

Reference:
https://github.com/youtube-jocoding/pyupbit-autotrade/tree/main

'''

import time
import pyupbit
import datetime
import numpy as np

# These are the keys you get from Upbit API
access = "ak"
secret = "sk"

# Returns the price of when to buy
def get_target_buying_price(ticker, k):
    print("Target Buying Price Retrieved")
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    target_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return target_price

# Returns the start time of the market
def get_start_time(ticker):
    print("Starting Time Retrieved")
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0]
    return start_time

# Returns the current balance of the ticker
def get_balance(ticker):
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

# Returns the current price of the ticker
def get_current_price(ticker):
    print("Current Price Retrieved")
    return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"]

# Returns the moving average line of the past 14 days
def get_ma14(ticker):
    df = pyupbit.get_ohlcv(ticker, interval="day", count=14)
    if df is not None:
        ma14 = df['close'].rolling(13).mean().iloc[-1]
        print("MA14 is: " + str(ma14))
        return ma14
    else:
        print(f"No data for {ticker}")
        return None


# Returns ror of k
def get_ror(k):
    df = pyupbit.get_ohlcv("KRW-BTC", count=7)
    df['range'] = (df['high'] - df['low']) * k
    df['target'] = df['open'] + df['range'].shift(1)

    # 0.05 수수료
    df['ror'] = np.where(df['high'] > df['target'],
                         df['close'] / df['target'] - 0.05,
                         1)

    ror = df['ror'].cumprod()[-2]
    return ror

# Returns the value of k that gives the best ror, using get_ror
# according to the last 7 days' data
def get_bestk():
    t = {}
    for k in np.arange(0.1, 1.0, 0.1):
        ror = get_ror(k)
        t[k] = ror
    maxk = max(t, key = t.get)
    return maxk

# Belowe code buys crypto according to the target price and sells by the end of one market day (9AM)

# Log in with your access and secret key
upbit = pyupbit.Upbit(access, secret)
print("START TRADING")

while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-BTC")
        end_time = start_time + datetime.timedelta(days=1)

        if start_time < now < end_time - datetime.timedelta(seconds=30):
            k = get_bestk()
            print("Best k is: " + str(k))
            target_bp = get_target_buying_price("KRW-BTC", k)
            current_price = get_current_price("KRW-BTC")
            ma14 = get_ma14("KRW-BTC")
            if target_bp < current_price and ma14 < current_price:
                krw = get_balance("KRW")
                # krw has to be more that 5000, as the least amount you can trade is 5000 won
                if krw > 5000:
                    upbit.buy_market_order("KRW-BTC", krw*0.9995)
                    print("BTC bought")
        else:
            btc = get_balance("BTC")
            # btc has to be more than 0.00008, which is about 5000 won
            if btc > 0.00008:
                upbit.sell_market_order("KRW-BTC", btc*0.9995)
                print("BTC sold")
                break
        time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(1)

print("END of Trade")
