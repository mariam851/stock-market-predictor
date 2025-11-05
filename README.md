# Stock Market Predictor – LSTM Model

![Project Preview](screenshots/preview.jpeg)

## Overview
**Stock Market Predictor** is a web app that forecasts future stock prices using a pre-trained **LSTM (Long Short-Term Memory) model**.  
It allows users to:  

- Visualize historical stock prices.  
- Compare predicted prices with actual prices.  
- Forecast the next 30 trading days.  
- Gain actionable investment insights.  

**Technologies Used:**  
Python, Keras, Streamlit, Yahoo Finance API (`yfinance`), Matplotlib

---

## Features
- Input **any stock symbol** (e.g., `AAPL`, `GOOG`, `TSLA`).  
- Select **start date** for historical analysis.  
- Display historical stock prices in a **data table**.  
- Predict stock prices for **test data** and **next 30 trading days**.  
- Unified graph showing **original**, **predicted**, and **future** prices.  
- Generate **investment insights**:
  - Last available price  
  - Highest & lowest prices  
  - Average price & standard deviation  
  - Expected price & expected change after 30 days  

---

## How It Works
1. Enter stock symbol and choose a start date.  
2. Fetch historical stock data from **Yahoo Finance**.  
3. Split data into **train (80%)** and **test (20%)** sets.  
4. Scale data for LSTM input.  
5. Predict prices for test set and the next 30 trading days.  
6. Display **graphical trends** and predictions.  
7. Show **investment insights** for decision-making.  

---

## Investment Insights Example

| Metric | Value |
|--------|-------|
| Last Available Price | 180.38 USD |
| Highest Price in Test Period | 180.38 USD |
| Lowest Price in Test Period | 18.82 USD |
| Average Price in Test Period | 67.51 USD |
| Price Std Dev in Test Period | 31.06 USD |
| Expected Price after 30 Days | 128.65 USD |
| Expected Average Price Next 30 Days | 142.45 USD |
| Expected Price Std Dev Next 30 Days | 10.65 USD |
| Expected Change in 30 Days | -51.73 USD (-28.68%) |

> **Note:** Predictions are based on historical patterns. Market events and news are **not included**.

---

## How to Run Locally

```bash
git clone https://github.com/mariam851/stock-market-predictor.git
cd stock-market-predictor
pip install -r requirements.txt
streamlit run app.py
