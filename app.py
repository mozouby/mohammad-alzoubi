import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import plotly.graph_objects as go


# -----------------------------
# Page Settings
# -----------------------------

st.set_page_config(
    page_title="AI Finance Dashboard",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 AI Finance Dashboard")
st.write(
    "AI powered stock analysis and prediction platform"
)


# -----------------------------
# Sidebar
# -----------------------------

symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    "NVDA"
)


period = st.sidebar.selectbox(
    "Select Data Period",
    [
        "6mo",
        "1y",
        "2y",
        "5y"
    ]
)


run = st.sidebar.button(
    "Analyze Stock"
)



# -----------------------------
# Main Application
# -----------------------------

if run:

    with st.spinner("Loading financial data..."):


        data = yf.download(
            symbol,
            period=period
        )


    if data.empty:

        st.error(
            "Stock symbol not found"
        )

        st.stop()



    data = data.reset_index()



    # -----------------------------
    # Price Chart
    # -----------------------------

    st.subheader(
        "📈 Historical Stock Price"
    )


    chart = go.Figure()


    chart.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines",
            name="Price"
        )
    )


    chart.update_layout(
        height=450
    )


    st.plotly_chart(
        chart,
        use_container_width=True
    )



    # -----------------------------
    # Market Information
    # -----------------------------

    current_price = float(
        data["Close"].iloc[-1]
    )


    highest = float(
        data["High"].max()
    )


    lowest = float(
        data["Low"].min()
    )



    col1,col2,col3 = st.columns(3)


    col1.metric(
        "Current Price",
        f"${current_price:.2f}"
    )


    col2.metric(
        "Highest Price",
        f"${highest:.2f}"
    )


    col3.metric(
        "Lowest Price",
        f"${lowest:.2f}"
    )



    # -----------------------------
    # Machine Learning Model
    # -----------------------------


    st.subheader(
        "🧠 AI Price Prediction"
    )


    data["Days"] = np.arange(
        len(data)
    )


    X = data[
        ["Days"]
    ]


    y = data[
        "Close"
    ]



    split = int(
        len(data)*0.8
    )


    X_train = X[:split]
    y_train = y[:split]


    X_test = X[split:]
    y_test = y[split:]



    model = LinearRegression()


    model.fit(
        X_train,
        y_train
    )



    prediction_test = model.predict(
        X_test
    )


    error = mean_absolute_error(
        y_test,
        prediction_test
    )



    future_day = np.array(
        [[len(data)+30]]
    )


    future_prediction = model.predict(
        future_day
    )[0]



    col4,col5 = st.columns(2)


    col4.metric(
        "30 Day Prediction",
        f"${future_prediction:.2f}"
    )


    col5.metric(
        "Model Error",
        f"{error:.2f}"
    )



    # -----------------------------
    # Data Table
    # -----------------------------


    st.subheader(
        "📊 Latest Market Data"
    )


    st.dataframe(
        data.tail(20),
        use_container_width=True
    )



    # -----------------------------
    # Download Data
    # -----------------------------


    csv = data.to_csv(
        index=False
    )


    st.download_button(
        "Download CSV",
        csv,
        "stock_data.csv"
    )



else:

    st.info(
        "Enter a stock symbol and click Analyze Stock"
    )
