# Trading Bot for Upbit

## Files
- Strategy Trading
- LSTM Trading
- GNB Trading
- Informer Trading
- Prophet Trading

## Introduction
This repository focuses on exploring multiple algorithms, machine learning models and strategies that can be used in forecasting
time series data, which can be applied to stock or crypto currency market predictions.
Currently, the main focus is to apply Long Short Term Memory (LSTM), Gaussian Naive Bayes (GNB), Informer (based on transformer)
to predict crypto currency price or return using Upbit API, in which Upbit is the largest coin exchange market in South Korea in trading volume. 

## Focus and Purpose
Many sources from the Internet describe how to use LSTM model to predict stock/crypto price. However, transformer has replaced many models including
LSTM and RNN, which makes using LSTM feel outdated. Consequently, I will be researching more on how to apply transformer, informer particularly, on
predicting the market. I want to focus on exploring informer suggested in the paper "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", as it 
suggests improvements on original transformer model and apply it to time series data for forecasting.
In addition to exploring existing algorithms and models, I have also tried to create trading strategy of my own, and testing it on the market.

## Informer
I wanted to build a system that analyzes and forecasts multiple crypto currencies in order to invest in the currency with the highest potential predicted value. While I was constructing and experimenting this system with LSTM model, I have came across a problem of slow training time. In order to analyze and forecast multiple currencies, the model has to learn each currency’s time series price data. Using LSTM took too long and was not efficient. Consequently, I looked for another model that could be used and came across the paper “Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting” by Haoyi Zhou et al. This paper suggests a model called Informer to enhance efficiency on time series forecasting. 

There are some limitations on using LSTM and transformer for prediction in time series data. Firstly, the quadratic computation of self attention makes the time complexity O(n^2). Additionally, there is also a memory bottleneck in stacking layers for long inputs. Moreover, there is a speed plunge in the prediction of long outputs in the step by step decoding. There have been several suggestions of new models based on transformer to solve the such problems. For example, the model reformer is only specialized for extremely long input sequence. Likewise, models suggested focus on improving one weakness of transformer out of multiple. On the other hand, the informer model is suggested to improve on all three weaknesses that I have mentioned above. 

First of all, informer suggests that while using transformer based model, it proposes ProbSparse self attention mechanism to reduce time complexity and memory usage to O(nlogn). Also, it proposes generative style decoder, which lets the model to have long sequence output with only one forward step, preventing cumulative error that can be caused in the step by step inference. 
The paper defines Long Sequence Time-series Forecasting (LSTF) to have an output length of 48 or greater. 
