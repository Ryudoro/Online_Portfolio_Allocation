import yfinance as yf
import matplotlib.pyplot as plt

def rolling_mean(df):
    return df.rolling(window=20).mean().dropna()

def rolling_std(df):
    return df.rolling(window=20).std().dropna()
    
def plot(df):
    plt.scatter(df.index, df['Close'], s = 1)
    
def bollinger(df):
    return rolling_mean(df) - 2 * rolling_std(df), rolling_mean(df)+2 * rolling_std(df)

def checking_bollinger(stock, period):
    data = yf.download(stock, period)
    bollinger_down, bollinger_up = bollinger(data)
    
    data_close = data['Close']
    bollinger_close_down, bollinger_close_up = bollinger_down['Close'], bollinger_up['Close']

def main():
    data = yf.download('AAPL', period= '1y')
    print(data.head())
    plot(data)


# data = yf.download('AAPL', period = '1y')
# a,b = bollinger(data)
# # main()
# # plt.show()

# # print(a)
# # print(b)

# plt.plot(a.index, a['Close'])
# plt.plot(b.index, b['Close'])
# plt.plot(data.index, data['Close'])

# plt.show()