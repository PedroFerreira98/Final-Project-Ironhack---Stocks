import pandas as pd
from collect_data import stock_values_connector as stock_prices_class


#Get All values from stock daily
stock_values = stock_prices_class.get_stock_values('MSFT')
stock_values['Ticker'] = 'MSFT'
stock_values.to_csv('stock_values.csv')

#Get annual income statement from stock
annual_income_statement = stock_prices_class.annual_income_statement('MSFT')

#Get financial ratios from stock
financial_ratios = stock_prices_class.annual_financial_ratios('MSFT')


#Join financial tables
frames = [annual_income_statement, financial_ratios]
financial_dataframe = pd.concat(frames)
financial_dataframe = financial_dataframe.T
financial_dataframe['Ticker'] = 'MSFT'
financial_dataframe.to_csv('financial_ratios.csv')

