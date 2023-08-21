# from Stats.statistics import main
# import matplotlib.pyplot as plt
# main()
# plt.show()
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

days_for_training = 500
days_for_testing = 0
days_in_future = 50
name_of_company = 'ALO.PA'
name_of_model = 'trained_model_'+name_of_company.replace('.','')+'.h5'
#data, data_to_use = search_input(name_of_company)

data = yf.download(name_of_company, period= '5y')
#data = pd.DataFrame(data)
# Choix de la colonne à prédire
target_column = 'Close'  
data_to_use = data[target_column].values


# Préparation des données pour l'entraînement
scaler = MinMaxScaler(feature_range=(0,1))

#change ligne -> colonne + minmaxscaler
scaled_data = scaler.fit_transform(data_to_use.reshape(-1, 1))

x_train, y_train = [], []

#Boucle d'entrainement avec i qui pars du nombre de jour d'entrainement jusqu'au nombre de donnée - le nombre de jour de test, le temps est découpé
#en 0, day_for_training, scaled_data-days_for_testing, scaled_data
for i in range(days_for_training, len(scaled_data)-days_for_testing):
    # print(f'X_train for {i} is ',scaled_data[i-days_for_training:i, 0])
    # print(f'y_train for {i} is ',scaled_data[i, 0])
    # print('lenght of X_train is', len(scaled_data[i-days_for_training:i, 0]))
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

if len(X_test) != 0:
    last_days_for_input = X_test[-1]
else:
    last_days_for_input = x_train[-1]
    
loaded_model = load_model('trained_model_ALOPA.h5')

prediction_list = last_days_for_input

for _ in range(days_in_future):
    x = prediction_list[-days_for_training:]
    x = x.reshape((1, days_for_training, 1))
    out = loaded_model.predict(x)[0][0]
    prediction_list = np.append(prediction_list, out)

prediction_list = prediction_list[days_for_training-1:]

# Inverse the normalization
predicted_future = scaler.inverse_transform(prediction_list.reshape(-1, 1))

plt.plot(predicted_future)
plt.show()