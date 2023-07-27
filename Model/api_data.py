import yfinance as yf

def get_stock_data(stock_symbol, period):
    # Récupérer les données du stock
    stock = yf.Ticker(stock_symbol)

    # Récupérer les données historiques
    hist = stock.history(period=period)

    # Retourner les données
    return hist

# # Utiliser la fonction
# data = get_stock_data("AAPL", "1y")  # Récupère les données d'Apple sur un an
# print(data)