from input_creation import search_input
from model_creation import model_creation
from model_compilation import model_compilation
from futur_prediction import load_model_and_predict
from model_data_creation import model_data_creation
from futur_prediction import predict_future
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
import sys
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


global_dir = '/home/supersymmetry/Documents/Online_Portfolio_Allocation'
sys.path.append(global_dir)
from Stats.statistics import bollinger
days_for_training = 500
days_for_testing = 100
days_in_future = 7
name_of_company = 'ALO.PA'
#name_of_model = 'trained_model_'+name_of_company.replace('.','')+'.h5'
name_of_model = 'test.h5'
name_of_model = 'Model_stock/trained_model_ALOPA.h5'
data, data_to_use = search_input(name_of_company, jenkins = False)
x_train, y_train, X_test, scaler = model_data_creation(data_to_use, days_for_training, days_for_testing)
# model = model_creation(x_train, y_train)
# joblib.dump(scaler, 'scaler.pkl')
# model.save(name_of_model)

# if not os.path.exists(name_of_model):
#     model_compilation(x_train, y_train, name_of_model)
model = load_model(name_of_model)
if len(X_test) != 0:
    last_days_for_input = X_test[-1]
else:
    last_days_for_input = x_train[-1]

scaler = joblib.load('Model_stock/scalerALOPA.pkl')


if len(X_test) != 0:
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    predicted_dates = data.index[-len(predicted_price):]
    
plt.figure(figsize = (12,6))

if len(X_test) != 0:
    plt.plot(data.index[:len(data)-len(predicted_price)], data['Close'][:len(data)-len(predicted_price)], color ='blue', label="prix réel jusqu'au moment du test")
    plt.plot(predicted_dates, predicted_price, color = 'red', label = 'prix prédit')
    plt.plot(predicted_dates, data['Close'][len(data)-len(predicted_price):], color = 'green', label = "prix réel au moment du test")
    
    
plt.xlabel("Date")
plt.ylabel("Prix (euros)")
plt.legend()

plt.title("Evolution réel et prédiction du cours de l'action ALO.PA")

if len(X_test) != 0:
    rmse = np.sqrt(mean_squared_error(data['Close'][len(data)-len(predicted_price):], predicted_price))
    mae = mean_absolute_error(data['Close'][len(data)-len(predicted_price):], predicted_price)
    r2 = r2_score(data['Close'][len(data)-len(predicted_price):], predicted_price)

    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R-squared: {r2:.2f}')
    
plt.show()


# print("real value is: ", last_days_for_input[-5])
# print("len of real value is: ", len(last_days_for_input))
# print("type of real value: ", type(last_days_for_input))
# print("shape of real value is: ", last_days_for_input.shape)
predicted_future = predict_future(days_in_future, model, last_days_for_input, scaler, days_for_training)
#predicted_future = load_model_and_predict(name_of_model, days_in_future, last_days_for_input, scaler, days_for_training)
# print("last value is ", last_days_for_input)
# print("predicted value is ", predicted_future)
last_date = data.index[-1]
# print(scaler.inverse_transform(last_days_for_input[-5:].reshape(-1,1)))

# print(model.summary())
# print(scaler.get_params())
# boll_down, boll_up = bollinger(pd.Series(last_days_for_input.reshape(-1)))

# print(boll_down.iloc[-1])
# plt.plot(boll_down)
# plt.plot(boll_up)
# # Create a new date index for the future predictions
# date_index_future = pd.date_range(start=last_date, periods=days_in_future+1)  # Adding 1 because date_range is exclusive of the endpoint

# # Plotting data
# #plt.plot(date_index_future[1:], predicted_future[:-1], color='green', label='Prix prédit')  # Starting from index 1 because date_index_future includes the last_date

# plt.show()