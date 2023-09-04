from futur_prediction import load_model_and_predict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def test_result(name_of_model, last_days_for_input, scaler, days_for_training):
    
    #result = load_json_model_and_predict(model, 1, last_days_for_input, scaler, days_for_training)
    predicted_future = load_model_and_predict(name_of_model, 1, last_days_for_input, scaler, days_for_training)
    
    boll_down, boll_up = bollinger(pd.Series(last_days_for_input.reshape(-1)))
    
    boll_down = scaler.inverse_transform(np.array(boll_down).reshape(-1, 1)).reshape(-1)
    boll_up = scaler.inverse_transform(np.array(boll_up).reshape(-1, 1)).reshape(-1)
    if (float(predicted_future[-1]) > float(boll_down[-1])) and (float(predicted_future[-1]) < float(boll_up[-1])):
        return True
    else:
        raise ValueError
    
def rolling_mean(df):
    return df.rolling(window=20).mean().dropna()

def rolling_std(df):
    return df.rolling(window=20).std().dropna()
    
def plot(df):
    plt.scatter(df.index, df['Close'], s = 1)
    
def bollinger(df):
    return rolling_mean(df) - 2 * rolling_std(df), rolling_mean(df)+2 * rolling_std(df)

if __name__ == '__main__':
    from model_data_creation import model_data_creation
    from input_creation import search_input
    
    data, data_to_use = search_input()
    days_for_training = 100
    days_for_testing = 0
    x_train, y_train, X_test, scaler = model_data_creation(data_to_use, days_for_training, days_for_testing)
    last_days_for_input = x_train[-1]
    
    test_result('Model_stock/trained_model_ALOPA.h5', last_days_for_input, scaler, days_for_training)
    
    