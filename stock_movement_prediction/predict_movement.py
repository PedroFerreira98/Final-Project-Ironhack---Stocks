from collect_data import stock_values_connector as stock_prices_class
#from sklearn.preprocessing import MinMaxScaler
#import math
#import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split
#from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
#from datetime import timedelta
#from business_calendar import Calendar
from pandas.tseries.offsets import BDay
from datetime import datetime, timedelta, date
from fbprophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima






def check_accuracy(real_close, predicted_close):
    len_real_close = real_close.shape[0]
    len_predicted_close = predicted_close.shape[0]

    first_value_index = len_real_close - len_predicted_close - 1 #In order to get one value before the predicted list
    second_value_index = len_real_close - len_predicted_close #In order to get first value in predicted list

    predicted_movement_correctly = 0
    for index in range(len_predicted_close):
        print('Real close + index:', real_close[first_value_index + index])
        print('Real close + index, fist in real:', real_close[second_value_index+index])
        print('Predicted close , index:', predicted_close[index])
        if (real_close[first_value_index + index] - real_close[second_value_index+index] > 0) and (real_close[first_value_index + index] - predicted_close[index] > 0):
            predicted_movement_correctly += 1
        elif (real_close[first_value_index + index] - real_close[second_value_index+index] < 0) and (real_close[first_value_index + index] - predicted_close[index] < 0):
            predicted_movement_correctly += 1
        elif (real_close[first_value_index + index] - real_close[second_value_index+index] == 0) and (real_close[first_value_index + index] - predicted_close[index] == 0):
            predicted_movement_correctly += 1
        else:
            predicted_movement_correctly += 0
    return predicted_movement_correctly/len_predicted_close

def check_accuracy_percentage(real_close, predicted_close): # Its correct if its around 6% the real value
    len_real_close = real_close.shape[0]
    len_predicted_close = predicted_close.shape[0]

    first_value_index = len_real_close - len_predicted_close  # In order to get first value in predicted list

    inside_interval = 0
    for index in range(len_predicted_close):
        print(first_value_index)
        print(real_close[first_value_index+index])
        print(predicted_close[index])
        if (real_close[first_value_index+index]-real_close[first_value_index+index]*0.06) <= predicted_close[index] <= (real_close[first_value_index+index]+real_close[first_value_index+index]*0.06):
            inside_interval += 1
        else:
            inside_interval += 0
    return inside_interval/len_predicted_close

stock_prices = stock_prices_class.stock_close('MSFT')

print(len(stock_prices)*0.2)


# Predicting with Long Short-Term Memory Model

#Split Data in 80% training
training_set = stock_prices[:1006]
test_set = stock_prices[1006:]




#Normalize data, might increase performance
scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(training_set)

#Creating data structure with 60 time steps and 1 output
X_train = []
y_train = []
for i in range(60, 1006):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Build Model
model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units=50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units=1))
# Compiling the RNN (Recurrent Neural Network)
model.compile(optimizer='adam', loss='mean_squared_error')
# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs=25, batch_size=32)



dataset_train = pd.DataFrame(training_set)
dataset_test = pd.DataFrame(test_set)



dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
#inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = dataset_total[60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
print(inputs.shape)



X_test = []
for i in range(60, 1198):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)



#Predict using test set
a1 = X_test
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


stock_values = stock_prices_class.get_stock_values('MSFT')
stock_values = stock_values['Close']


#Predict  using ARIMA model
from statsmodels.tsa.ar_model import AR
train_data, test_data = stock_values[0:int(len(stock_values)*0.90)], stock_values[int(len(stock_values)*0.90):]
# Create model
model = AR(train_data)

# Train model
model_fit = model.fit(maxlag=1) #use previous 10 predicted values to predict next value

# Prediction
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)

r2_score(test_data, predictions)


'''
model_autoARIMA = auto_arima(train_data,
                             start_p=0,
                             start_q=0,
                             test='adf',
                             max_p=3,
                             max_q=3,
                             m=1,
                             d=None,
                             seasonal=False,
                             start_P=0,
                             D=0,
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,stepwise=True)

print(model_autoARIMA.summary())
model = ARIMA(train_data, order=(3, 1, 3))
fitted = model.fit(disp=-1)
fc, se, conf = fitted.forecast(544, alpha=0.05)  # 95% confidence
fc_series = pd.Series(fc, index=test_data.index)
'''
stock_values = stock_prices_class.get_stock_values('MSFT')
#Predict using Prophet
data_prophet = stock_values[['Date', 'Close']]
data_prophet = data_prophet.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(daily_seasonality=True) # the Prophet class (model)
m.fit(data_prophet) # fit the model using all data

future = m.make_future_dataframe(periods=365) #we need to specify the number of days in future
prediction = m.predict(future)
fig1, ax1 = plt.subplots()
m.plot(prediction)
plt.title("Prediction MSFT Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()



accuracy_by_movement = check_accuracy(stock_prices, predicted_stock_price)
accuracy_by_interval = check_accuracy_percentage(stock_prices, predicted_stock_price)




future_days = []
#yesterday = (date.today() + BDay(1))
today = (date.today() + BDay(0))
stock_values = stock_values.append({'Date': today}, ignore_index=True)






data = stock_values
fig, ax = plt.subplots()
ax.plot(stock_values.Date[120:1258], stock_values.Close[120:1258], color='blue', label='Real MSFT values')
ax.plot(stock_values.Date[120:1258], predicted_stock_price, color='red', label='Predicted MSFT values')
fmt = mdates.MonthLocator(interval=3)
ax.xaxis.set_major_locator(fmt)
plt.xticks(rotation=45)

