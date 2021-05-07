from collect_data import stock_values_connector as stock_prices_class
from datetime import datetime, timedelta, date
from pandas.tseries.offsets import BDay
from fbprophet import Prophet

def prophet_prediction(ticker):
    stock_values = stock_prices_class.get_stock_values(ticker)
    len_original = len(stock_values)
    #Predict using Prophet
    data_prophet = stock_values[['Date', 'Close']]
    data_prophet = data_prophet.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet(daily_seasonality=True) # the Prophet class (model)
    m.fit(data_prophet) # fit the model using all data

    future = m.make_future_dataframe(periods=15) #we need to specify the number of days in future
    prediction = m.predict(future)
    prediction = prediction.trend[-14:].values

    for day in range(1, 15):
        today = (date.today() + BDay(day))
        stock_values = stock_values.append({'Date': today}, ignore_index=True)

    close = stock_values.Close.values
    close_prices = []
    i = 0
    for element in range(len(stock_values)):
        if element < len_original:
            close_prices.append(close[element])
        elif i < 14:
            close_prices.append(prediction[i])
            i+=1


    stock_values.Close = close_prices
    return stock_values