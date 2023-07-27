import numpy as np
from tensorflow.keras.models import load_model

def predict_future(days_in_future, model, last_days_for_input, scaler, days_for_training):
    prediction_list = last_days_for_input

    for _ in range(days_in_future):
        x = prediction_list[-days_for_training:]
        x = x.reshape((1, days_for_training, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    
    prediction_list = prediction_list[days_for_training-1:]

    # Inverse the normalization
    predicted_future = scaler.inverse_transform(prediction_list.reshape(-1, 1))

    return predicted_future

def load_model_and_predict(model_file, days_in_future, last_days_for_input, scaler, days_for_training):
    # Load the model
    loaded_model = load_model(model_file)

    prediction_list = last_days_for_input

    for _ in range(days_in_future):
        x = prediction_list[-days_for_training:]
        x = x.reshape((1, days_for_training, 1))
        out = loaded_model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    
    prediction_list = prediction_list[days_for_training-1:]

    # Inverse the normalization
    predicted_future = scaler.inverse_transform(prediction_list.reshape(-1, 1))

    return predicted_future