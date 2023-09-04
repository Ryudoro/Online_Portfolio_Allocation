from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def model_creation(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=32,activation = 'relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    #model.save(name_of_model)
    return model

if __name__ == '__main__':
    import numpy as np
    from model_data_creation import model_data_creation
    from input_creation import search_input
    
    data, data_to_use = search_input()
    days_for_training = 500
    days_for_testing = 30
    x_train, y_train, X_test, scaler = model_data_creation(data_to_use, days_for_training, days_for_testing)

    model = model_creation(x_train, y_train)