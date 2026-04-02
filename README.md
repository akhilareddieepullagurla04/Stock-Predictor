import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📈 Stock Price Predictor")

# Input
stock = st.text_input("Enter Stock Symbol", "AAPL")

# Load Data with caching (VERY IMPORTANT)
@st.cache_data
def load_data(symbol):
    try:
        data = yf.Ticker(symbol).history(period="3y")
        return data
    except:
        return pd.DataFrame()

# Button to avoid auto-loading issues
if st.button("Load Data"):

    with st.spinner("Fetching stock data... ⏳"):
        data = load_data(stock)

    # Check data
    if data is None or data.empty:
        st.error("❌ Failed to fetch data. Try another stock (AAPL, TSLA, INFY.NS)")
    else:
        st.success("✅ Data Loaded Successfully")

        # Show data
        st.subheader("📊 Recent Data")
        st.dataframe(data.tail())

        # Chart
        st.subheader("📈 Closing Price Chart")
        st.line_chart(data["Close"])

        # ML Model
        data = data[['Close']].copy()
        data['Prediction'] = data['Close'].shift(-10)

        data.dropna(inplace=True)

        X = np.array(data[['Close']])
        y = np.array(data['Prediction'])

        model = LinearRegression()
        model.fit(X, y)

        # Predict next 10 days
        future = np.array(data[['Close']].tail(10))
        predictions = model.predict(future)

        st.subheader("🔮 Next 10 Days Prediction")
        st.write(predictions)