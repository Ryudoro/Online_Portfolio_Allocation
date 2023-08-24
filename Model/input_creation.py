

class InvalidInputError(Exception):
    pass

stocks_list = ['GOOGL', 'AAPL', 'ALO.PA']
def search_input(stock_symbol = 'ALO.PA', period = '5y', jenkins = True):
    # Récupérer les données du stock
    is_valid = is_input_valid(stock_symbol, period)
    if not is_valid:
        return None, None
    if jenkins == True:
        import pandas as pd
        data = pd.read_csv('ALO.csv')
    else:
        import yfinance as yf
        data = yf.download(stock_symbol, period= period)
    #data = pd.DataFrame(data)
    # Choix de la colonne à prédire
    target_column = 'Close'  
    data_to_use = data[target_column].values

    return data, data_to_use


def is_input_valid(stock_symbol, period):
    if not (period.endswith('y') or period.endswith('m') or period.endswith('d')):
        return False
    if not period[:-1].isdigit():
        return False
    if stock_symbol not in stocks_list:
        return False
    if int(period[:-1]) > 5:
        return False
    return True

data, data_to_use = search_input()
