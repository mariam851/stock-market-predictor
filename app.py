import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler


model = load_model(r'C:\Users\MASTER\OneDrive\Desktop\Projects\Portfolio-Projects\predict stock prices\stock_predictions_model.keras')


st.set_page_config(page_title='Stock Market Prediction', layout='wide')


page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    background-image: 
        linear-gradient(0deg, transparent 24%, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.03) 26%, transparent 27%, transparent 74%, rgba(255,255,255,0.03) 75%, rgba(255,255,255,0.03) 76%, transparent 77%, transparent),
        linear-gradient(90deg, transparent 24%, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.03) 26%, transparent 27%, transparent 74%, rgba(255,255,255,0.03) 75%, rgba(255,255,255,0.03) 76%, transparent 77%, transparent);
    background-size: 50px 50px;
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: rgba(0, 0, 0, 0);
}

.block-container {
    background-color: rgba(22, 27, 34, 0.9);
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

h1, h2, h3, h4, h5, h6, label, p, span {
    color: #e6edf3 !important;
    font-family: 'Segoe UI', sans-serif;
}

input, textarea {
    background-color: #161b22 !important;
    color: white !important;
    border: 1px solid #30363d !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'> Stock Market Predictor</h1>", unsafe_allow_html=True)
st.write("### Predict future stock prices using a trained LSTM model")


stock = st.text_input('Enter Stock Symbol (e.g. AAPL, GOOG, TSLA)', 'GOOG')


start = '2012-01-01'
end = '2022-12-31'
data = yf.download(stock, start, end)

st.subheader('Stock Data (2012–2022)')
st.dataframe(data.tail(10))


data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


def styled_plot(title, *series):
    fig = plt.figure(figsize=(8, 5))
    for s in series:
        plt.plot(s['data'], s.get('style', '-'), label=s['label'])
    plt.title(title, color='white', fontsize=13)
    plt.xlabel('Time', color='white')
    plt.ylabel('Price', color='white')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#0d1117')
    plt.tick_params(colors='white')
    st.pyplot(fig)


ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

st.subheader('Price Trends & Moving Averages')
col1, col2 = st.columns(2)

with col1:
    styled_plot('Price vs MA50',
        {'data': ma_50_days, 'label': 'MA50', 'style': 'r'},
        {'data': data.Close, 'label': 'Price', 'style': 'g'}
    )

with col2:
    styled_plot('Price vs MA100 vs MA200',
        {'data': ma_100_days, 'label': 'MA100', 'style': 'b'},
        {'data': ma_200_days, 'label': 'MA200', 'style': 'y'},
        {'data': data.Close, 'label': 'Price', 'style': 'g'}
    )


x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)
predict = model.predict(x)

scale = 1/scaler.scale_
predict = predict * scale
y = y * scale


st.subheader('Original vs Predicted Prices')
fig4 = plt.figure(figsize=(10, 5))
plt.plot(predict, 'r', label='Predicted')
plt.plot(y, 'g', label='Original')
plt.xlabel('Time', color='white')
plt.ylabel('Price', color='white')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().set_facecolor('#0d1117')
plt.tick_params(colors='white')
st.pyplot(fig4)

st.write(" **Model Loaded Successfully** — Ready to Predict Future Trends!")
