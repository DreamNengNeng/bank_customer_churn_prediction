
"""
File: Predict Stock Price by using Linear Regression.py
@author: Haoyu Wang
"""

#Step 0. Import the libraries

import time
import datetime
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pandas_ta as ta

#Step 1. Extract historical stock data from finance.yahoo.com

tickers = ['AAPL']   #'TSLA','TWTR', 'GOOG','MSFT','AMZN'     #Choose your interested stock's ticker then input here
period1 = int(time.mktime(datetime.datetime(2020,1,1,23,59).timetuple()))    #Time period can be changed per your interest
period2 = int(time.mktime(datetime.datetime(2021,11,30,23,59).timetuple()))
interval = '1d'                                                              #interval can be adjusted as needed. Here we choose '1d' since this program is to help daily trade. 

xlwriter = pd.ExcelWriter('AAPL historical 2020-2021 Stock price.xlsx', engine = 'openpyxl')
for ticker in tickers:
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    df.to_excel(xlwriter, sheet_name=ticker, index = False)
    
xlwriter.save()


#show some summary statistics
print(df.info())
df1 = df[['Date','Open','Close']]
print(df1)

# Step 2. Clean the data
#Reindex data using a datetime Index

df1.set_index(pd.DatetimeIndex(df['Date']), inplace=True ) 
sns.lineplot(data=df1[['Date','Close','Open']])
 # Notice the plot may appear BEHIND other windows

# Linear regression assumption: variables in the data are independent, so we need to use technical indicaters like exponential moving average (EMA)

df1.ta.ema(close='Close', length = 10, append=True)

print(df1.head(30))

#Drop the first n-rows   
#df = df.dropna()
df1 = df1.iloc[9:]

print(df1.head(10))


sns.lineplot(data=df1)

xlwriter = pd.ExcelWriter('cleanded stock data with EMA_10.xlsx', engine = 'openpyxl')
df.to_excel(xlwriter, sheet_name=ticker, index = False)
xlwriter.save()

#Step 3. Build the model
from sklearn import datasets, linear_model 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1[['EMA_10']], df1['Close'], test_size =.2)

print(X_train.describe()) 
print(X_test)  
print(y_train) 
print(X_train.describe())
print(y_test.describe())
print(y_test) 
print(y_train.describe())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# create Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

X_test.shape
#(93, 1)
y_pred.shape
#(93,)
y_test.shape

# Printout relevant statisticl metrics
print("Model Coefficients:", model.coef_)
print("Model Intercept", model.intercept_)
print('Mean squared error (MSE) : %.2f' 
      % mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination (R^2): %.2f" 
      % r2_score(y_test, y_pred))

# a lower MAE value is better, and the closer our coefficient of the correlation value is to 1.0 the better

df2 = pd.DataFrame({'Actual Close': y_test, 'Predicted Close': y_pred})

sns.lineplot(data=df2)

df3 = df2.merge(df1,left_index = True, right_index = True)
print(df3)

df3['Actual Gain'] = df3['Actual Close'] - df3['Open']
df3['Predicted Gain'] = df3['Predicted Close']- df3['Open']
print(df3.info())
print(df3)

xlwriter = pd.ExcelWriter('Stock Prediction based on Daily Stock Closing Price and EMA.xlsx', engine = 'openpyxl')
df3.to_excel(xlwriter, sheet_name = ticker, index=True)
xlwriter.save()

