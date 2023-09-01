from tensorflow.keras.models import Sequential

def model_compilation(x_train, y_train, name_of_model):
    model = Sequential()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    model.save(name_of_model)
    
x_train = [[[0.        ],
  [0.11111111],
  [0.22222222],
  [0.33333333],
  [0.44444444]],

 [[0.11111111],
  [0.22222222],
  [0.33333333],
  [0.44444444],
  [0.55555556]]]

y_train = [0.55555556, 0.66666667]

model_compilation(x_train, y_train, 'billy.h5')