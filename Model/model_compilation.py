from tensorflow.keras.models import Sequential

def model_compilation(x_train, y_train, name_of_model):
    model = Sequential()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    model.save(name_of_model)