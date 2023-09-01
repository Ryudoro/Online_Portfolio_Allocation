from Model.futur_prediction import load_json_model_and_predict

def test_result(model, last_days_for_input, scaler, days_for_training):
    
    result = load_json_model_and_predict(model, 1, last_days_for_input, scaler, days_for_training)
    
    