'''
This code is solely designed by Joon Song, myself. I have tried to
create code that analyzes which crypto currency has the most upward
movement. If it detects a sudden upward movement, it buys the coin and 
sells at given percentage return. This code automatically chooses the currency
with most upward movement and tries to make profit from the movement. 
However, sometimes the price falls after sudden increase, so I have created counter
so that if the price doesn't make profit, it sells before too much loss.

© Written by Joon Song
'''

import time
import pyupbit
import datetime
import numpy as np

# These are the keys you get from Upbit API
access = "a"
secret = "s"

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

def abv_avg_vol(tickers):
    ticker_num = 0
    total = 0
    tickas = []
    temp = {}
    for t in tickers:
        df = pyupbit.get_ohlcv(t, interval= 'day', count= 1)
        if df is None:
            continue   
        temp[t] = df['value'][0]
        total += df['value'][0]
        ticker_num += 1
    avg = (total / ticker_num) * 1.7
    # print(avg)
    for t, v in temp.items():
        if v > avg:
            tickas.append(t)
    return tickas

def get_upward():
    all_tickers = pyupbit.get_tickers(fiat="KRW")
    tickers = abv_avg_vol(all_tickers)
    print(tickers)
    difs = {}
    for t in tickers:
        # special case
        if t == "KRW-REP":
            continue
        df = pyupbit.get_ohlcv(t, interval= 'minute1', count=4)
        if df is None:
            continue
        difs[t] = df['close'][3] / df['close'][0]
       
    maxup = max(difs, key=difs.get)
    if difs[maxup] <= 1.0015:
        return 'NO'
    print(maxup + ': '+ str(difs[maxup]))
    return maxup

def get_ma(ticker):
    df = pyupbit.get_ohlcv(ticker, interval="minute1", count=12)
    if df is not None:
        ma14 = df['close'].rolling(11).mean().iloc[-1]
        return ma14
    else:
        print(f"No data for {ticker}")
        return None
    

upbit = pyupbit.Upbit(access, secret)
print("START TRADING")
bought = 0
bp = 0
ticker = None
hold_count = 0
loss = 0
gain = 0

while True:
    try:
        if bought == 0:
            ticker = get_upward()
        if ticker == 'NO':
            continue
        current_price = pyupbit.get_current_price(ticker)
        ma = get_ma(ticker)
        if bought == 0:
            krw = get_balance("KRW")
            if krw > 5000 and current_price <= ma:
                upbit.buy_market_order(ticker, krw*0.9995)
                print("MA is: " + str(ma))
                print("BOUGHT at: " + str(current_price))
                bp = current_price
                bought = 1
        if bought == 1:
            currency = ticker.split("-")[1]
            coin = get_balance(currency)
            if hold_count > 10 and current_price/bp >= 1.0045:
                upbit.sell_market_order(ticker, coin*0.9995)
                print(currency + ': ' + str(coin))
                print("SOLD with profit at: "+ str(current_price))
                bp = 0
                bought = 0
                hold_count = 0
                gain += 1
            elif hold_count > 2000 and current_price/bp > 1.0015:
                upbit.sell_market_order(ticker, coin*0.9995)
                print(ticker + str(coin))
                print("SOLD with minimum profit of " + str(current_price/bp)+"at: "+ str(current_price))
                hold_count = 0
                bp = 0
                bought = 0
                gain += 1
            elif hold_count > 4000 and current_price/bp <= 1.0015:
                upbit.sell_market_order(ticker, coin*0.9995)
                print(ticker + str(coin))
                print("SOLD with loss at: "+ str(current_price))
                hold_count = 0
                bp = 0
                bought = 0
                loss += 1
                break
            else:
                hold_count += 1
                if hold_count % 100 == 0:
                    print(hold_count)
        
        if gain == 10:
            break
    
    except Exception as e:
        print(e)
        time.sleep(1)
print(gain)
print(loss)
print("END")