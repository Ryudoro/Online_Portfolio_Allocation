from input_creation import search_input
from model_creation import model_creation
from model_compilation import model_compilation
from futur_prediction import load_model_and_predict
from model_data_creation import model_data_creation
import pandas as pd
import os
import matplotlib.pyplot as plt

days_for_training = 500
days_for_testing = 0
days_in_future = 50
name_of_company = 'ALO.PA'
name_of_model = 'trained_model_'+name_of_company.replace('.','')+'.h5'
data, data_to_use = search_input(name_of_company)
x_train, y_train, X_test, scaler = model_data_creation(data_to_use)
model = model_creation(x_train)

if not os.path.exists(name_of_model):
    model_compilation(x_train, y_train, name_of_model)

if len(X_test) != 0:
    last_days_for_input = X_test[-1]
else:
    last_days_for_input = x_train[-1]

predicted_future = load_model_and_predict(name_of_model, days_in_future, last_days_for_input, scaler, days_for_training)

last_date = data.index[-1]

# Create a new date index for the future predictions
date_index_future = pd.date_range(start=last_date, periods=days_in_future+1)  # Adding 1 because date_range is exclusive of the endpoint

# Plotting data
plt.plot(date_index_future[1:], predicted_future[:-1], color='green', label='Prix prédit')  # Starting from index 1 because date_index_future includes the last_date

plt.show()