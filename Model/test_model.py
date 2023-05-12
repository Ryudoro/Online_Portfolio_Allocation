import alpaca_trade_api as tradeapi
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import time

# API
API_KEY = 'api_key'
API_SECRET = 'api_secret'
BASE_URL = 'https://paper-api.alpaca.markets'  # URL demo

# Initialisation
api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL)

# trade
symbol = 'AAPL'

def trade(symbol):
    # données du marché
    barset = api.get_barset(symbol, 'day', limit=21)
    bars = barset[symbol]

    # Préparation modèle ARIMA
    data = pd.DataFrame([bar.c for bar in bars], columns=['close'])
    data.index = pd.DatetimeIndex([bar.t for bar in bars])

    # Entraîne le modèle ARIMA
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit(disp=0)

    # Prédit le prix de clôture du lendemain
    forecast = model_fit.forecast(steps=1)[0]

    # Si le prix prédit est supérieur au prix actuel, achète l'action
    if forecast > data['close'][-1]:
        # Récupère le nombre d'action
        account = api.get_account()
        cash = float(account.cash)
        price = bars[-1].c
        shares = cash // price

        # Passe l'ordre d'achat
        if shares > 0:
            print(f"Achat de {shares} actions de {symbol} à {price}")
            api.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='gtc'
            )

# Boucle et trade chaque jour
while True:
    trade(symbol)

    # Attend le jour suivant
    time.sleep(86400)
