from sqlalchemy import create_engine
import pandas as pd
from collect_data import stock_values_connector as stock_prices_class
from stock_movement_prediction import prophet_prediction as prophet_predict
import mysql.connector as m

ticker='MSFT'

def update_mysql(ticker):
    #Get All values from stock daily
    stock_values = prophet_predict.prophet_prediction(ticker)
    stock_values['Ticker'] = ticker
    #stock_values.to_csv('stock_values.csv')

    #Get annual income statement from stock
    annual_income_statement = stock_prices_class.annual_income_statement(ticker)

    #Get financial ratios from stock
    financial_ratios = stock_prices_class.annual_financial_ratios(ticker)


    #Join financial tables
    frames = [annual_income_statement, financial_ratios]
    financial_dataframe = pd.concat(frames)
    financial_dataframe = financial_dataframe.reset_index()
    financial_dataframe = financial_dataframe.rename(columns={'index': 'Indicator'})
    financial_dataframe = pd.melt(financial_dataframe, id_vars=['Indicator'], value_vars=['2016','2017','2018','2019','2020'])
    financial_dataframe = financial_dataframe.rename(columns={'value': 'Indicator_value','variable':'Year'})
    indexNames = financial_dataframe[financial_dataframe['Indicator'] == 'reportedCurrency'].index
    indexNames1 = financial_dataframe[financial_dataframe['Indicator'] == 'fillingDate'].index
    indexNames2 = financial_dataframe[financial_dataframe['Indicator'] == 'acceptedDate'].index
    indexNames3 = financial_dataframe[financial_dataframe['Indicator'] == 'period'].index
    indexNames4 = financial_dataframe[financial_dataframe['Indicator'] == 'link'].index
    indexNames5 = financial_dataframe[financial_dataframe['Indicator'] == 'finalLink'].index
    financial_dataframe.drop(indexNames, inplace=True)
    financial_dataframe.drop(indexNames1, inplace=True)
    financial_dataframe.drop(indexNames2, inplace=True)
    financial_dataframe.drop(indexNames3, inplace=True)
    financial_dataframe.drop(indexNames4, inplace=True)
    financial_dataframe.drop(indexNames5, inplace=True)
    financial_dataframe['Ticker'] = ticker
    #financial_dataframe.to_csv('financial_ratios.csv')
    financial_dataframe['Year'] = financial_dataframe['Year'].astype('datetime64[ns]')
    financial_dataframe['Indicator_value'] = financial_dataframe['Indicator_value'].astype('float')

    #Get balance sheet
    balance_sheet = stock_prices_class.annual_balance_sheet(ticker)
    balance_sheet = balance_sheet.reset_index()
    balance_sheet = balance_sheet.rename(columns={'index': 'Indicator'})
    balance_sheet = pd.melt(balance_sheet, id_vars=['Indicator'], value_vars=['2016','2017','2018','2019','2020'])
    balance_sheet = balance_sheet.rename(columns={'value': 'Indicator_value','variable':'Year'})
    indexNames = balance_sheet[balance_sheet['Indicator'] == 'reportedCurrency'].index
    indexNames1 = balance_sheet[balance_sheet['Indicator'] == 'fillingDate'].index
    indexNames2 = balance_sheet[balance_sheet['Indicator'] == 'acceptedDate'].index
    indexNames3 = balance_sheet[balance_sheet['Indicator'] == 'period'].index
    indexNames4 = balance_sheet[balance_sheet['Indicator'] == 'link'].index
    indexNames5 = balance_sheet[balance_sheet['Indicator'] == 'finalLink'].index
    balance_sheet.drop(indexNames, inplace=True)
    balance_sheet.drop(indexNames1, inplace=True)
    balance_sheet.drop(indexNames2, inplace=True)
    balance_sheet.drop(indexNames3, inplace=True)
    balance_sheet.drop(indexNames4, inplace=True)
    balance_sheet.drop(indexNames5, inplace=True)
    balance_sheet['Ticker'] = ticker
    #balance_sheet.to_csv('balance_sheet.csv')
    balance_sheet['Year'] = balance_sheet['Year'].astype('datetime64[ns]')
    balance_sheet['Indicator_value'] = balance_sheet['Indicator_value'].astype('float')

    #Get Cash Flow
    cash_flow = stock_prices_class.annual_cash_flow(ticker)
    cash_flow = cash_flow.reset_index()
    cash_flow = cash_flow.rename(columns={'index': 'Indicator'})
    cash_flow = pd.melt(cash_flow, id_vars=['Indicator'], value_vars=['2016','2017','2018','2019','2020'])
    cash_flow = cash_flow.rename(columns={'value': 'Indicator_value','variable':'Year'})
    indexNames = cash_flow[cash_flow['Indicator'] == 'reportedCurrency'].index
    indexNames1 = cash_flow[cash_flow['Indicator'] == 'fillingDate'].index
    indexNames2 = cash_flow[cash_flow['Indicator'] == 'acceptedDate'].index
    indexNames3 = cash_flow[cash_flow['Indicator'] == 'period'].index
    indexNames4 = cash_flow[cash_flow['Indicator'] == 'link'].index
    indexNames5 = cash_flow[cash_flow['Indicator'] == 'finalLink'].index
    cash_flow.drop(indexNames, inplace=True)
    cash_flow.drop(indexNames1, inplace=True)
    cash_flow.drop(indexNames2, inplace=True)
    cash_flow.drop(indexNames3, inplace=True)
    cash_flow.drop(indexNames4, inplace=True)
    cash_flow.drop(indexNames5, inplace=True)
    cash_flow['Ticker'] = ticker
    #cash_flow.to_csv('cash_flow.csv')
    cash_flow['Year'] = cash_flow['Year'].astype('datetime64[ns]')
    cash_flow['Indicator_value'] = cash_flow['Indicator_value'].astype('float')



    engine = create_engine("mysql+pymysql://{user}:{pw}@localhost:3306/{db}"
                       .format(user="root",
                               pw="Portugal12",
                               db="stock_user_database"))

    stock_values.to_sql('stock_values', engine, index=False, if_exists="replace")
    financial_dataframe.to_sql('financial_ratios', engine, index=False, if_exists="replace")
    balance_sheet.to_sql('balance_sheet', engine, index=False, if_exists="replace")
    cash_flow.to_sql('cash_flow', engine, index=False, if_exists="replace")


    # this part of the connector simply opens the door for the connection to the database
    '''
    pass_sql = 'Portugal12'  # please write your SQL pass here
    cnx = m.connect(user='root', password=pass_sql,host='localhost',
                    database='stock_user_database',
                    auth_plugin='mysql_native_password')

    # this line of code is always a good practice!!
    if cnx.is_connected():
        print("Connection open")
        # do stuff you need to the database
    else:
        print("Connection is not successfully open")

    
    cursor = cnx.cursor()

    query = "SELECT project_sql.city.city_name,project_sql.country.country_name, project_sql.country.GDP_Education, s.school FROM project_sql.country INNER JOIN project_sql.city ON project_sql.country.Country_Id = project_sql.city.Country_Id INNER JOIN project_sql.city_school ON project_sql.city.city_id = project_sql.city_school.city_id INNER JOIN project_sql.schools as s ON project_sql.city_school.schools_id = s.schools_id ORDER BY project_sql.country.GDP_Education DESC LIMIT 6;"

    cursor.execute(query)
    question_table = pd.DataFrame(cursor.fetchall())

    cnx.close()
    '''