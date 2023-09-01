import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# days_for_training = 500
# days_for_testing = 1

def model_data_creation(data_to_use, days_for_training = 100, days_for_testing =0):

    # Préparation des données pour l'entraînement
    scaler = MinMaxScaler(feature_range=(0,1))
    

    is_valid = is_data_valid(data_to_use, days_for_training = 100, days_for_testing =0)
    if not is_valid:
        return None, None, None, scaler
    
    scaled_data = scaler.fit_transform(data_to_use.reshape(-1, 1))

    x_train, y_train = [], []
    
    for i in range(days_for_training, len(scaled_data)-days_for_testing):
        x_train.append(scaled_data[i-days_for_training:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Préparation des données pour les tests
    inputs = data_to_use[len(data_to_use) - days_for_testing - days_for_training:]
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test = []
    if len(inputs) > days_for_training:
        for i in range(days_for_training, len(inputs)):
            X_test.append(inputs[i-days_for_training:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    return x_train, y_train, X_test, scaler

def is_data_valid(data_to_use, days_for_training, days_for_testing):
    if len(data_to_use) - days_for_testing - days_for_training < 0:
        return False
    if len(data_to_use) == 0:
        return False
    if days_for_training == 0:
        return False
    return True


data_to_use = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
days_for_training = 5
days_for_testing = 3
x_train, y_train, X_test, scaler = model_data_creation(data_to_use, days_for_training, days_for_testing)
