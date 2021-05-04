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

#Get balance sheet
balance_sheet = stock_prices_class.annual_balance_sheet('MSFT')
balance_sheet = balance_sheet.T
balance_sheet['Ticker'] = 'MSFT'
balance_sheet.to_csv('balance_sheet.csv')

#Get Cash Flow
cash_flow = stock_prices_class.annual_cash_flow('MSFT')
cash_flow = cash_flow.T
cash_flow['Ticker'] = 'MSFT'
cash_flow.to_csv('cash_flow.csv')
