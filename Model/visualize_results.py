import matplotlib.pyplot as plt
from futur_prediction import load_model_and_predict
from input_creation import search_input
import pandas as pd
import joblib

def visualize_result(name_of_company, predicted_future, period = '5y', days_in_future = 10):
    
    name_of_model = 'trained_model_'+name_of_company.replace('.','')+'.h5'
    data, data_to_use = search_input(name_of_company)
    days_for_training = 500
    days_for_testing = 0
    
    scaler = joblib.load('scaler.pkl')
    input = data_to_use[len(data_to_use) - days_for_testing - days_for_training:]
    input = input.reshape(-1,1)
    input = scaler.transform(input)
    
    last_days_for_input = input
    
    predicted_future = load_model_and_predict(name_of_model, days_in_future, last_days_for_input, scaler, days_for_training)
    
    last_date = data.index[-1]

    # Create a new date index for the future predictions
    date_index_future = pd.date_range(start=last_date, periods=days_in_future+1)  # Adding 1 because date_range is exclusive of the endpoint
    print(predicted_future)
    # Plotting data
    plt.plot(date_index_future[1:], predicted_future[:-1], color='green', label='Prix prédit')  # Starting from index 1 because date_index_future includes the last_date

    plt.show()
    
visualize_result('ALO.PA', 1)