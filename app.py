import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pandas.tseries.offsets import BDay  

# Load trained LSTM model
model = load_model(r'C:\Users\MASTER\OneDrive\Desktop\Projects\Portfolio-Projects\predict stock prices\stock_predictions_model.keras')

# Streamlit page config
st.set_page_config(page_title='Stock Market Predictor', layout='wide')

# Page styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
.block-container { background-color: rgba(22, 27, 34, 0.9); border-radius: 15px; padding: 2rem; box-shadow: 0 0 20px rgba(0,0,0,0.4);}
h1, h2, h3, label, p, span { color: #e6edf3 !important; font-family: 'Segoe UI', sans-serif;}
input, textarea { background-color: #161b22 !important; color: white !important; border: 1px solid #30363d !important;}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>Stock Market Predictor</h1>", unsafe_allow_html=True)
st.write("### Predict future stock prices using a trained LSTM model")

# User input: stock symbol
stock = st.text_input('Enter Stock Symbol (e.g. AAPL, GOOG, TSLA)', 'GOOG')

# User input: start date
st.subheader('Select the Start Date for Analysis')
start_date = st.date_input('Start Date', value=datetime(2012, 1, 1))

# Download data and automatically set end date to last available
temp_data = yf.download(stock, start=start_date, end=datetime.today())
end_date = temp_data.index[-1].date()
st.write(f"End Date automatically set to latest available: {end_date}")

data = yf.download(stock, start=start_date, end=end_date)
st.subheader(f'Stock Data ({start_date} – {end_date})')
st.dataframe(data.tail(10))

# Split Train/Test
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_full)

# Prepare LSTM input
x_test, y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Predict Test Set
y_pred = model.predict(x_test)
scale = 1/scaler.scale_
y_test_scaled = np.array(y_test) * scale
y_pred_scaled = y_pred * scale

# Dates for Test Set
dates_test = data.index[int(len(data)*0.80):]

# Predict next 30 trading days
last_100_days = data_test_scaled[-100:]
future_predictions = []
future_dates = pd.bdate_range(start=data.index[-1] + BDay(1), periods=30)
current_input = last_100_days.copy()

for i in range(30):
    current_input_reshaped = np.reshape(current_input, (1, current_input.shape[0], 1))
    pred = model.predict(current_input_reshaped)[0,0]
    future_predictions.append(float(pred * scale))
    current_input = np.append(current_input[1:], [[pred]], axis=0)

# -----------------------
# Unified Plot
# -----------------------
st.subheader('Stock Price Overview & Future Predictions')
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(dates_test, y_test_scaled, label='Original Price', color='green')
ax.plot(dates_test, y_pred_scaled, label='Predicted Price (Test Set)', color='red')
ax.plot(future_dates, future_predictions, label='Future Predictions (30 Days)', color='blue', linestyle='--')
ax.set_xlabel('Date (X axis)', color='white')
ax.set_ylabel('Price (Y axis)', color='white')
ax.set_title(f'{stock} Stock Price & Predictions', color='white')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_facecolor('#0d1117')
ax.tick_params(colors='white', rotation=45)
st.pyplot(fig)

# -----------------------
# Investment Insights
# -----------------------
last_actual_price = float(y_test_scaled[-1])
max_price = float(np.max(y_test_scaled))
min_price = float(np.min(y_test_scaled))
mean_price = float(np.mean(y_test_scaled))
std_price = float(np.std(y_test_scaled))

future_change = future_predictions[-1] - last_actual_price
future_change_pct = (future_change / last_actual_price) * 100
future_mean = np.mean(future_predictions)
future_std = np.std(future_predictions)

st.subheader('Investment Insights')
st.markdown(f"""
- **Last Available Price:** {last_actual_price:.2f} USD  
- **Highest Price in Test Period:** {max_price:.2f} USD  
- **Lowest Price in Test Period:** {min_price:.2f} USD  
- **Average Price in Test Period:** {mean_price:.2f} USD  
- **Price Std Dev in Test Period:** {std_price:.2f} USD  

- **Expected Price after 30 Days:** {future_predictions[-1]:.2f} USD  
- **Expected Average Price Next 30 Days:** {future_mean:.2f} USD  
- **Expected Price Std Dev Next 30 Days:** {future_std:.2f} USD  
- **Expected Change in 30 Days:** {future_change:.2f} USD ({future_change_pct:.2f}%)  
""")

# -----------------------
# Future Predictions Table (Daily)
# -----------------------
st.subheader('Daily Future Predictions (Next 30 Trading Days)')

# Format dates without leading zeros
future_df = pd.DataFrame({
    'Date': [f"{d.day}-{d.month}-{d.year}" for d in future_dates],
    'Predicted Price (USD)': future_predictions
})
st.dataframe(future_df.style.format({'Predicted Price (USD)': "{:.2f}"}))
