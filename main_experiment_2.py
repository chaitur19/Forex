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
USDAUD_model = load_model("USDAUD_model.h5")
USDCAD_model = load_model("USDCAD_model.h5")
USDCNY_model = load_model("USDCNY_model.h5")
USDGBP_model = load_model("USDGBP_model.h5")

allmodels = {'USDINR': USDINR_model, 'USDAUD': USDAUD_model, 
'USDCAD':USDCAD_model, 'USDCNY':USDCNY_model, 'USDGBP':USDGBP_model }

currency_pair_list = ('USDINR', 'USDAUD', 'USDCAD', 'USDCNY', 'USDGBP')

n_steps = 100


def choose_dataset(currency_pair_list, allmodels):
    #st.sidebar.subheader('Select the forex pair')
    currencypair = st.sidebar.selectbox("**Select the forex pair**", currency_pair_list, key='1')
    st.markdown(
    """<style>
div[class*="stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {font-size: 19px;}
    </style>""", unsafe_allow_html=True)

    #st.sidebar.selectbox()
    # st.sidebar.write(check)
    for itr in currency_pair_list:
        if currencypair == itr:
            main_df = pdr.get_data_yahoo(itr + '=X', start='2010-01-01', interval='1D')
            main_df.reset_index(inplace=True)
            main_df.drop('Volume', axis=1, inplace=True)
            model = allmodels[itr]
    return main_df, currencypair, model

def sentiment_analyis_prediction(currencypair, currency_pair_list):

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    import pandas as pd
    import time
    import csv

    # initializing chrome web driver
    #driver = webdriver.Chrome(executable_path='chromedriver_win32/chromedriver.exe')
    
    @st.experimental_singleton
    def get_driver():
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')

    driver = get_driver()
    driver.get("http://example.com")
    st.write(driver.page_source)

    #st.code(driver.page_source)

    #dummy=input("Input currency forex pair")
    currency="INR"

    searchList=['Politics','Congress','GDP','CPI','Gold+price','Employment','Parliament','Banks','Democrat','Bear+Market','Bull+run','NATO','G7','War','G20','Silicon+valley','Natural+gas+Price','White+House','Stock+market','Capital+market','Inflation','Oil+prices','Natural+Calamities','BJP','B20','Stress+testing']
    searchList.insert(0, currency)

    data=[]
    for i in searchList:
        
        #resp=driver.get('https://www.google.com/search?q='+i+'&authuser=0&tbm=nws&sxsrf=APwXEddvFjyX-E3Jv6SMNq5CWpM-X9jFSg:1681868177846&ei=kUU_ZJOXM8_cptQPkoOQmA8&start=20&sa=N&ved=2ahUKEwjTo7Ka57T-AhVProkEHZIBBPM4ChDy0wN6BAgFEAc&biw=1036&bih=909&dpr=1.02')
        #driver.get('http://www.example.com')
        dv = driver.find_elements("xpath", '//div[@class="SoaBEf"]')
        for element in dv:
            title= element.find_element("xpath", './/div[@class="mCBkyc ynAwRc MBeuO nDgy9d"]')
            link = element.find_element("xpath", './/div/a')
            des = element.find_element("xpath", './/div[@class="GI74Re nDgy9d"]')
            date = element.find_element("xpath", './/div[@class="OSrXXb ZE0LJd YsWzw"]')
            publisher=element.find_element("xpath", './/div[@class="CEMjEf NUnG9d"]')

            data.append([title.text, link.get_attribute('href'), des.text, date.text, publisher.text])

    #driver.close()

    df=pd.DataFrame(data, columns=['Titles', 'Links', 'Details', 'Date', 'Publisher'])
    #st.write(df)
    # print(df.head())
    df.to_csv('GoogleNews.csv')

    df1=df
    from datetime import date, datetime
    today = datetime.today().strftime('%Y-%m-%d')

    
    import datetime as dt

    df1['Date'] = df1['Date'].apply(lambda x: dt.datetime.today().strftime('%Y-%m-%d') if 'hour' in x or 'min' in x else (dt.datetime.today() - dt.timedelta(days=1)).strftime('%Y-%m-%d') if 'day ago' in x else (dt.datetime.today() - dt.timedelta(days=2)).strftime('%Y-%m-%d') if '2 days ago' in x else (dt.datetime.today() - dt.timedelta(days=3)).strftime('%Y-%m-%d') if '3 days ago' in x else (dt.datetime.today() - dt.timedelta(days=4)).strftime('%Y-%m-%d') if '4 days ago' in x else (dt.datetime.today() - dt.timedelta(days=5)).strftime('%Y-%m-%d') if '5 days ago' in x else (dt.datetime.today() - dt.timedelta(days=6)).strftime('%Y-%m-%d') if '6 days ago' in x else (dt.datetime.today() - dt.timedelta(days=7)).strftime('%Y-%m-%d') if '1 week ago' in x else x)
    df1 = df1[~df1['Date'].str.contains('2022')]
    df1 = df1[~df1['Date'].str.contains('weeks ago')]
    df1 = df1[~df1['Date'].str.contains('month ago')]

    df1 = df1.reset_index(drop=True)

    df1 = df1[df1['Date'].str.contains('\d{4}-\d{2}-\d{2}')] # keep rows with date in YYYY-MM-DD format
    df1['Date'].unique()

    # assuming df1 is your dataframe,converting the 'Date' column from string to datetime format
    df1['Date'] = pd.to_datetime(df1['Date'], infer_datetime_format=True)


    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    #!pip install langid
    import langid

    # Download the Vader lexicon and langid model
    nltk.download('vader_lexicon')
    langid.set_languages(['en', 'hi', 'fr', 'de', 'es', 'it', 'ja', 'ko', 'nl', 'pt', 'ru', 'zh', 'ar'])

    # Filter out non-English titles using langid
    #st.write(df1.columns)
    df1['lang'] = df1['Titles'].apply(lambda x: langid.classify(x)[0])
    df1 = df1[df1['lang'] == 'en']


    # Extract the 'Title' column and convert it to a list
    titles = df1['Titles'].tolist()

    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Create an empty list to hold the sentiment scores
    sentiment_scores = []
    polarity=[]

    # Perform sentiment analysis on each title
    for title in titles:
        # Analyze the sentiment of the title using Vader lexicon
        sentiment = analyzer.polarity_scores(title)

        sentiment_scores.append(sentiment['compound'])
        
        # Determine the sentiment label based on the compound score
        if sentiment['compound'] > 0:
            sentiment_label = 'positive'
        elif sentiment['compound'] < 0:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Append the sentiment label to the list
        polarity.append(sentiment_label)

    # Add the sentiment scores to the dataframe as a new column
    df1['sentiment_value'] = sentiment_scores
    df1['polarity'] = polarity

    

    # Output the modified dataframe to a new CSV file
    df1.drop(['Links', 'Publisher', 'lang'], axis=1, inplace=True)
    df1=df1[['Date','Titles', 'sentiment_value','polarity']]

    st.subheader('News Data and Sentiment')
    st.write(df1)
    #df1.to_csv('sentiment_scores.csv', index=False)

     # Calculate sentiment score for each date
    grouped = df1.groupby('Date')

    sentiment_by_date = grouped.agg({'Titles': 'count', 'sentiment_value': 'mean'})

    sentiment_by_date.drop(['Titles'], axis=1, inplace=True)

    #st.write(sentiment_by_date)


    for itr in currency_pair_list:
        if currencypair == itr:
            #data = yf.download(tickers='USDCAD=X', period='max', interval='1d')
            data = pdr.get_data_yahoo(itr + '=X', start='2010-01-01', interval='1D')
            data.drop('Volume', axis=1, inplace=True)
            st.subheader("Daily forex Data")
            #data.merge(sentiment_by_date)
            data2 = pd.merge(data, sentiment_by_date, on='Date')
            st.write(data2.tail(10))
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
            st.subheader("Predictions")
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
    df2=df.copy()
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


    ###########################################################################################
    ###################################### 30 days ############################################

    df=df2
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

    num_prediction = 30
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)
    forecast = forecast.reshape(1, -1)
    forecast = scaler.inverse_transform(forecast)
    #forecast
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

    st.subheader("Forecasted Monthly Data")
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)
    # fig.show()



    #############################################################################################

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="close"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def data_visualization(data,currency, graph):
        plt.figure(figsize=[10,6])
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

    plt.figure(figsize=(12,6))
    ax = sns.boxplot(x=monthly_data['Year'],y=monthly_data[graph],palette='RdBu')
    ax.set_title('Box Plots Year Wise')
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
        "**Menu**",key="visibility",options=["Home", "Data Visualization", "Numerical Prediction", "Sentiment Prediction"])

    st.markdown(
    """<style>
div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {font-size: 19px;}
    </style>""", unsafe_allow_html=True)

    temp, name, model = choose_dataset(currency_pair_list, allmodels)
    


    
    if radio_option == 'Home':
        
        st.markdown("<h3 style='text-align: left; style= color: black;'>Forex Price Predictor</h3>", unsafe_allow_html=True)
        st.write("")
        image = Image.open('forex1.png')
        st.image(image, width=700)
        st.write("**At Forex Price Predictor**, we understand the importance of reliable data and accurate predictions in the world of forex trading. Whether you're a seasoned trader or just getting started, our mission is to provide the most up-to-date and comprehensive forex price predictions you need to succeed.")
        st.write(
            """
        Below are the forex services provided by this app : 
            
        - Data Insights
        - Forex prediction using macro economic factors
        - Forex prediction using latest news and respective polarities
                     
        """)

        st.sidebar.subheader("Live currency exchange")
        def scrape_currency():
            import datetime
            import streamlit as st
            import pandas as pd
            import plotly.express as px
            import datetime
            import requests
            from bs4 import BeautifulSoup
            #import fbprophet
            #from fbprophet import Prophet
            from prophet import Prophet
            from prophet.plot import plot_plotly
            today = datetime.date.today()

            base_url = "https://www.x-rates.com/historical/?from=USD&amount=1&date"

            year = today.year
            month = today.month if today.month > 9 else f"0{today.month}"
            day = today.day if today.day > 9 else f"0{today.day}"
            day = day-1
            #st.write(day)

            URL = f"{base_url}={year}-{month}-{day}"
            #st.write(URL)
            page = requests.get(URL)
            #st.write(page)

            soup = BeautifulSoup(page.content, "html.parser")

            table = soup.find_all("tr")[12:]
            #st.write("table", table)

            currencies = [table[i].text.split("\n")[1:3][0] for i in range(len(table))]
            #st.write("currencies",currencies)
            currencies.insert(0, "date(y-m-d)")
            currencies.insert(1, "American Dollar")
            rates = [table[i].text.split("\n")[1:3][1] for i in range(len(table))]
            #st.write(rates)
            rates.insert(0, f"{year}-{month}-{day}")
            rates.insert(1, "1")
            curr_data = {currencies[i]: rates[i] for i in range(len(rates))}
            #st.write(curr_data)
            curr_data = pd.DataFrame(curr_data, index=[0])
            curr_data.to_csv("data.csv")
            cols = curr_data.columns

            return curr_data, cols[1:]


        daily_df, columns = scrape_currency()
        
        input = st.sidebar.number_input('Amount', value=1)
        base_curr = st.sidebar.selectbox("Select the base currency", columns)
        selected_curr = st.sidebar.multiselect("Select currencies", columns)
            #title = st.text_input('Enter Base Currency')
            #st.write(v, type(v))
            #st.write('The Base Currency is', title)
        if selected_curr:
                #title = st.text_input('Enter Base Currency')
            base = daily_df[base_curr].astype(float)
                #st.write(base)
            selected = daily_df[selected_curr].astype(float)
            #st.write(selected)
            calculated = selected.iloc[0][0]
            #st.write(calculated)
            converted = selected / float(base)
                #st.write(converted)
            calculated=calculated.astype(float)
                #st.write(type(calculated))
                #st.write(calculated)
            output=input*calculated
            st.sidebar.text_area(label="Output Data:", value=output, height=25)
            #st.write(input*calculated)



    if radio_option == 'Sentiment Prediction':

        sentiment_analyis_prediction(name, currency_pair_list)
        #st.plotly_chart(candle_stick_chart(temp), use_container_width=True)
        #m = Prophet()
        #m.fit(df)

    if radio_option == 'Data Visualization':
        

        st.subheader("**Data Visualization**")
        graph = st.selectbox("select plot",("Open", "High", "Low", "Close"))
        data_visualization(temp,name,graph)

        decomposition(temp,graph)

        macro = st.checkbox("**select to visualize macro economic data**", value=False)

        if macro:
            name
            part1 = "usd"

            part2 = (name[3:]).lower()
            if part2=='inr':
                part2='ind'
            data = pd.ExcelFile('monthly_prediction/compiled_data_new.xlsx')
            df = data.parse('Sheet1')
            df['m1_'+part2] = (df['m1_'+ part2 + '_rs'] / df['forex']) / 1000000000
            df['m1_'+part1] = df['m1_'+part1+'_b']

            df['cpi_d'] = df['cpi_'+part1] - df['cpi_'+part2]
            df['iip_d'] = df['iip_'+part1] - df['iip_'+part2]
            df['int_d'] = df['int_'+part1] - df['int_'+part2]
            df['m1_d'] = df['m1_'+part1+'_b'] - df['m1_'+part2]
            df['rsrv_d'] = df['reserves_'+part1] - df['reserves_'+part2]
            df['stock_d'] = df['stocks_'+part1] - df['stocks_'+part2]
            df['trade_d'] = df['trade_'+part1] - df['trade_'+part2]

            import datetime as dt

            df['month'] = pd.DatetimeIndex(df['date']).month
            df['year'] = pd.DatetimeIndex(df['date']).year

            df_d = df[['forex', 'cpi_d', 'iip_d', 'int_d', 'm1_d', 'rsrv_d', 'stock_d', 'trade_d']]

            y = df['forex']


            plt.figure(figsize=[10,6])
            #plt.title(graph+ ' Graph')
            plt.plot(df['date'], y)
            plt.title('Foreign Exchange Rate '+ name)
            plt.show()
            #st.pyplot(fig=plt)

            for params in ['cpi_', 'iip_', 'int_', 'm1_', 'reserves_', 'stocks_', 'trade_']:
                plt.figure(figsize=[10,6])
                if params in ['reserves_', 'trade_']:
                    plt.plot(df['date'], df[params+part2]/1000000000, label=name[3:])
                    plt.plot(df['date'], df[params+part1]/1000000000, label=name[0:3])
                else:
                    plt.plot(df['date'], df[params+part2], label='India')
                    plt.plot(df['date'], df[params+part1], label='USA')
                    
                if params == 'cpi_':
                    plt.title('Consumer Price Index (CPI)')
                elif params == 'iip_':
                    plt.title('Index of Industrial Production (IIP)')
                elif params == 'int_':
                    plt.title('Interest Rates (%)')
                elif params == 'm1_':
                    plt.title('Money Supply ($ Billion)')
                elif params == 'reserves_':
                    plt.title('Reserves ($ Billion)')
                elif params == 'stocks_':
                    plt.title('Stock Index (Scaled)')
                elif params == 'trade_':
                    plt.title('Net Exports ($ Billion)')
                plt.legend()
                plt.savefig(params + '.png')
                plt.show()
                st.pyplot(fig=plt)


    if radio_option == 'Numerical Prediction':

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
        forecast = st.slider("Days to forecast", min_value=1, max_value=10, step=2)
        st.subheader("Forecasted Daily data")

        plot_forecast_data(temp, forecast, model, name)


    
    #about_section()
