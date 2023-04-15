# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:14:48 2021
@author: 52pun
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import numpy as np
import base64
from pylab import rcParams
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose as sd
import seaborn as sns

yf.pdr_override()

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

USDINR_model = load_model("USDINR_model.h5")
#USDAUD_model = load_model("USDAUD_model.h5")
#USDCAD_model = load_model("USDCAD_model.h5")
#USDCNY_model = load_model("USDCNY_model.h5")
#USDGBP_model = load_model("USDGBP_model.h5")

allmodels = {'USDINR': USDINR_model}
#, 'USDAUD': USDAUD_model, 'USDCAD': USDCAD_model, 'USDCNY': USDCNY_model,
 #            'USDGBP': USDGBP_model}
currency_pair_list = ('USDINR', 'USDAUD', 'USDCAD', 'USDCNY', 'USDGBP')

n_steps = 100


def choose_dataset(currency_pair_list, allmodels):
    #st.sidebar.subheader('Select the forex pair')
    currencypair = st.sidebar.selectbox("**Select the forex pair**", currency_pair_list, key='1')
    st.markdown(
    """<style>
div[class*="stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {font-size: 19px;}
    </style>""", unsafe_allow_html=True)

    # st.sidebar.write(check)
    for itr in currency_pair_list:
        if currencypair == itr:
            main_df = pdr.get_data_yahoo(itr + '=X', start='2010-01-01', interval='1D')
            main_df.reset_index(inplace=True)
            main_df.drop('Volume', axis=1, inplace=True)
            model = allmodels[itr]
    return main_df, currencypair, model

def sentiment_analyis_prediction(currencypair, currency_pair_list):
    for itr in currency_pair_list:
        if currencypair == itr:
            #data = yf.download(tickers='USDCAD=X', period='max', interval='1d')
            data = pdr.get_data_yahoo(itr + '=X', start='2010-01-01', interval='1D')
            data.drop('Volume', axis=1, inplace=True)
            st.subheader("Daily forex Data")
            st.write(data.tail(10))
            #data_fb = data[data.index.year == pd.to_datetime("now").year]
            data_fb = data[data.index.year >= (pd.to_datetime("now").year-2)]
            #data_fb = data.copy()
            data_fb = data_fb.reset_index()
            data_fb = data_fb[['Date','Close']]

            df = pd.DataFrame()
            df['ds'] = pd.to_datetime(data_fb['Date'])
            df['y'] = data_fb['Close']
            df.head()

            m = Prophet()
            m.fit(df)

            future = m.make_future_dataframe(periods=6, freq='M')

            forecast = m.predict(future)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']].tail()
            fig3 = m.plot(forecast, xlabel='Date', ylabel='Value')
            st.subheader("Below is the prediction graph")
            st.write(fig3)
    #a = add_changepoints_to_plot(fig3.gca(), m, forecast)
    #candle_stick_chart(df)



def candle_stick_chart(data_ytd):
    # declare figure
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=data_ytd.index,
                                 open=data_ytd['Open'],
                                 high=data_ytd['High'],
                                 low=data_ytd['Low'],
                                 close=data_ytd['Close'], name='market data'))

    # Add titles
    fig.update_layout(
        title={
            'text': 'YTD Exchange Rates',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        autosize=False,
        width=1100,
        height=600,
        xaxis=go.layout.XAxis(linecolor='black',
                              linewidth=1,
                              mirror=True),

        yaxis=go.layout.YAxis(linecolor='black',
                              linewidth=1,
                              mirror=True),

        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4)
    )
    return fig

#st.subheader('USD to KES Exchange Rates Line Chart')
#st.caption("Line Charts are the conventional charts most commonly used for the "
 #          "fundamental and technical analysis to decide strategies.")

#data_ytd_l = data.reset_index()

def line_chart(data_ytd_l):
    for i in ['Open', 'High', 'Close', 'Low']:
        data_ytd_l[i] = data_ytd_l[i].astype('float64')

    fig = px.line(data_ytd_l, x="Date", y="Close")

    # Add titles
    fig.update_layout(
        title={
            'text': "All-time Close Rates",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        autosize=False,
        width=1100,
        height=600,
        xaxis=go.layout.XAxis(linecolor='black',
                              linewidth=1,
                              mirror=True),

        yaxis=go.layout.YAxis(linecolor='black',
                              linewidth=1,
                              mirror=True),

        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4)
    )

    return fig

    fig3 = m.plot(forecast)
    a = add_changepoints_to_plot(fig3.gca(), m, forecast)

def about_section():
    st.sidebar.subheader('Developed By:')
    st.sidebar.markdown("Team Super Six")

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def plot_predict(df, model, name):
    df = df.drop(["Open", "Low", "Adj Close"], axis=1)
    df = df.dropna()
    Date = df["Date"]
    close = df["Close"]
    close = close.dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    tmp = scaler.fit(np.array(close).reshape(-1, 1))
    new_df = scaler.transform(np.array(close).reshape(-1, 1))

    training_size = int(len(new_df) * 0.67)
    test_size = len(new_df) - training_size
    train_data, test_data = new_df[:training_size], new_df[training_size:]
    Date_train, Date_test = Date[:training_size], Date[training_size:]

    n_steps = 100
    time_step = n_steps
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    print(X_train.shape, X_test.shape)

    train_predict = model.predict(X_train)

    test_predict = model.predict(X_test)
    print(train_predict.shape, test_predict.shape)

    from sklearn.metrics import mean_squared_error
    print(f'Train error - {mean_squared_error(train_predict, Y_train) * 100}')
    print(f'Test error - {mean_squared_error(test_predict, Y_test) * 100}')

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    close_train = scaler.inverse_transform(train_data)
    close_test = scaler.inverse_transform(test_data)
    close_train = close_train.reshape(-1)
    close_test = close_test.reshape(-1)
    prediction = test_predict.reshape((-1))

    trace1 = go.Scatter(
        x=Date_train,
        y=close_train,
        mode='lines',
        name='Data'
    )
    trace2 = go.Scatter(
        x=Date_test[n_steps:],
        y=prediction,
        mode='lines',
        name='Prediction'
    )
    trace3 = go.Scatter(
        x=Date_test,
        y=close_test,
        mode='lines',
        name='Ground Truth'
    )
    layout = go.Layout(
        title=name,
        xaxis={'title': "Date"},
        yaxis={'title': "Close"}
    )
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

    st.plotly_chart(fig)
    # fig.show()


def plot_forecast_data(df, days, model, name):
    df = df.drop(["Open", "Low", "Adj Close"], axis=1)
    df = df.dropna()
    Date = df["Date"]
    close = df["Close"]
    close = close.dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    tmp = scaler.fit(np.array(close).reshape(-1, 1))
    new_df = scaler.transform(np.array(close).reshape(-1, 1))

    test_data = close
    test_data = scaler.transform(np.array(close).reshape(-1, 1))
    test_data = test_data.reshape((-1))

    def predict(num_prediction, model):
        prediction_list = test_data[-n_steps:]

        for _ in range(num_prediction):
            x = prediction_list[-n_steps:]
            x = x.reshape((1, n_steps, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[n_steps - 1:]

        return prediction_list

    def predict_dates(num_prediction):
        last_date = df['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
        return prediction_dates

    num_prediction = days
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)
    forecast = forecast.reshape(1, -1)
    forecast = scaler.inverse_transform(forecast)
    forecast
    test_data = test_data.reshape(1, -1)
    test_data = scaler.inverse_transform(test_data)
    test_data = test_data.reshape(-1)
    forecast = forecast.reshape(-1)
    res = dict(zip(forecast_dates, forecast))
    date = df["Date"]
    trace1 = go.Scatter(
        x=date,
        y=test_data,
        mode='lines',
        name='Data'
    )
    trace2 = go.Scatter(
        x=forecast_dates,
        y=forecast,
        mode='lines',
        name='Prediction'
    )
    layout = go.Layout(
        title=name,
        xaxis={'title': "Date"},
        yaxis={'title': "Close"}
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)
    # fig.show()
    choose_date = st.selectbox("Date", forecast_dates)
    for itr in res:
        if choose_date == itr:
            res_price = res[itr]
    st.write(f"On {choose_date} the forex price will be {res_price}")


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="close"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def data_visualization(data,currency, graph):
        plt.figure(figsize=[15,10])
        plt.title(graph+ ' Graph')
        plt.xlabel('Date')
        plt.ylabel(graph +' Price')
        plt.plot(data['Date'], data[graph]) 
        plt.show()
        st.pyplot(fig=plt)

def decomposition(temp,graph):
    data2 = temp
    data2.set_index('Date',inplace=True)
    monthly_mean = data2[graph].resample('M').mean()
    monthly_data = monthly_mean.to_frame()

    monthly_data['Year'] = monthly_data.index.year
    monthly_data['Month'] = monthly_data.index.strftime('%B')
    monthly_data['dayofweek'] = monthly_data.index.strftime('%A')
    monthly_data['quarter'] = monthly_data.index.quarter
    #monthly_data

    plt.figure(figsize=(12,12))
    ax = sns.boxplot(x=monthly_data['Year'],y=monthly_data[graph],palette='RdBu')
    ax.set_title('Box Plots Year Wise-Apple Stock Price')
    plt.style.context('fivethirtyeight')
    plt.show()
    st.pyplot(fig=plt)


    rcParams['figure.figsize'] = 26, 14
    plt.figure(figsize=[40,26])
    decomposed_series = sd(monthly_data[graph],model='additive')
    #plt.plot(decomposed_series)
    decomposed_series.plot()
    plt.show()
    st.pyplot(fig=plt)

    


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def landing_ui():
    st.header("Welcome to Forex Price Predictor")
    st.write("")
    st.write("")
    st.write("")
    st.write("Stay up-to-date with the latest forex market forecasts by following our analysis.\n we have used advanced algorithms and models to analyze market data and generate accurate predictions for currency exchange rates.\n Our website's prediction methodology is based on a combination of key market indicators, such as economic data releases, geopolitical events, and market sentiment, to develop a comprehensive view of the market and identify potential trading opportunities.\n We are constantly working to improve our models and algorithms, so that we can continue to deliver accurate predictions and help our visitors achieve their trading goals.\n Please note that Forex trading involves risks, and that our predictions are not guaranteed to be accurate. We encourage all visitors to carefully consider their trading strategies and to seek the advice of a qualified financial advisor before making any trades.\n Thank you for choosing our website for your Forex prediction needs.")
    st.write("")
    st.write("")

if __name__ == "__main__":

    image = Image.open('forex.png')

    side_bg = 'forex1.png'
    #sidebar_bg(side_bg)

    #add_bg_from_local('forex1.png')

    st.sidebar.image(image)#, caption='Sunrise by the mountains')
    st.sidebar.header("Forex Price Predictor")

    st.markdown(
    """<style>
div[class*="stSidebar"] > label > div[data-testid="stMarkdownContainer"] > p {font-size: 30px;}
    </style>""", unsafe_allow_html=True)

    st.sidebar.markdown("---")    

    #st.sidebar.subheader("Menu")
    radio_option = st.sidebar.radio(
        "**Menu**",key="visibility",options=["Home", "Data Visualization", "Numerical Predictions", "Sentiment based prediction"])

    st.markdown(
    """<style>
div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {font-size: 19px;}
    </style>""", unsafe_allow_html=True)

    temp, name, model = choose_dataset(currency_pair_list, allmodels)
    about_section()
    
    if radio_option == 'Home':
        
        
        # about_section()
        # print(temp)

        st.header("Welcome to Forex Price Predictor")
        st.write("")
        st.write("")
        st.write("")
        st.write("Welcome to Forex Price Predictor, your one-stop-platform for accurate and reliable forex price predictions.")
        st.write("At Forex Price Predictor, we understand the importance of reliable data and accurate predictions in the world of forex trading. Whether you're a seasoned trader or just getting started, our mission is to provide the most up-to-date and comprehensive forex price predictions you need to succeed.")
        st.write("we have used advanced algorithms and models to analyze market data and generate accurate predictions for currency exchange rates.")
        #st.write("Our website's prediction methodology is based on a combination of key market indicators, such as economic data releases, geopolitical events, and market sentiment, to develop a comprehensive view of the market and identify potential trading opportunities.")
        #st.write("We are constantly working to improve our models and algorithms, so that we can continue to deliver accurate predictions and help our visitors achieve their trading goals.")
        st.write("Please note that Forex trading involves risks, and that our predictions are not guaranteed to be accurate. We encourage all visitors to carefully consider their trading strategies and to seek the advice of a qualified financial advisor before making any trades.")
        st.write("Thank you for choosing our website for your Forex prediction needs.")
        st.write("")
        st.write("")

        #about_section()

    if radio_option == 'Sentiment based prediction':

        sentiment_analyis_prediction(name, currency_pair_list)
        #st.plotly_chart(candle_stick_chart(temp), use_container_width=True)
        #m = Prophet()
        #m.fit(df)

    if radio_option == 'Data Visualization':
        

        st.write("**Data Visualization**")
        graph = st.selectbox("select plot",("Open", "High", "Low", "Close"))
        data_visualization(temp,name,graph)

        decomposition(temp,graph)



    if radio_option == 'Numerical Predictions':

        st.header(f"Analyzing {name}'s foreign exchange data")
        st.write("")
        st.subheader("Daily forex Data")
        temp2=temp
        st.write(temp2.tail(10).style.hide_index())
        #st.write(temp.to_string(index=False))
        st.subheader("Forex Trend")
        plot_raw_data(temp)
        #st.plotly_chart(candle_stick_chart(temp), use_container_width=True)

        st.subheader("Model Performance")
        plot_predict(temp, model, name)

        #st.sidebar.subheader("Forecasted Data")
        #forecast_check = st.button("**Display Predictions**")
        #forecast_check = st.checkbox("**Display Predictions**", value=False)
        #about_section()
        #if forecast_check:
        forecast = st.slider("Days to forecast", min_value=10, max_value=30, step=5)
        st.subheader("Forecasted data")

        plot_forecast_data(temp, forecast, model, name)