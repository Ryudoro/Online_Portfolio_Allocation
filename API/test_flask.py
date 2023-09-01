# app.py
from flask import Flask, request, jsonify
import sys
import os
global_dir = '/home/project/Documents/Online_Portfolio_Allocation'
sys.path.append(global_dir)
from Model.input_creation import search_input
from Model.model_data_creation import model_data_creation
from Model.model_creation import model_creation
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import json

app = Flask("my API")

# Données de test
data = {
    "apple": "fruit",
    "banana": "fruit",
    "carrot": "vegetable",
    # Ajoutez plus de données ici
}

# Route pour récupérer les données en fonction de la requête de l'utilisateur
@app.route('/api/get_data', methods=['GET'])
def get_data():
    query = request.args.get('query')
    if query in data:
        return jsonify({query: data[query]})
    else:
        return jsonify({"message": "Données non trouvées."}), 404


@app.route('/api/get_stock_data', methods=['GET'])
def get_stock_data():
    
    stock_symbol = request.args.get('stock_symbol')
    period = '5y'
    jenkins = False
    data, data_to_use = search_input(stock_symbol, period, jenkins)
    
    if data is None or data_to_use is None:
        return jsonify({"error": "Invalid input"})
    
    data.reset_index(inplace=True) 
    
    result = {
        "stock_data": data.to_dict(),  # Convert DataFrame to dictionary
        "data_to_use": data_to_use.tolist()  # Convert numpy array to list
    }
    return jsonify(result)

@app.route('/api/prepare_data', methods=['POST'])
def prepare_data():

    #request_data = request.args.get()
    request_data = request.json
    data_to_use = np.array(request_data['data_to_use'])
    days_for_training = int(request_data['days_for_training'])
    days_for_testing = int(request_data['days_for_testing'])
    name_of_compagny = request_data['name_of_compagny']
    
    x_train, y_train, X_test, scaler = model_data_creation(data_to_use, days_for_training, days_for_testing)

    scaler_name = os.path.join(global_dir, f"scaler_{name_of_compagny.replace('.', '')}.pkl")
    joblib.dump(scaler, scaler_name)
    response = {
        'x_train': x_train.tolist(),
        'y_train': y_train.tolist(),
        'X_test': X_test,
    }

    return jsonify(response)

@app.route('/API/model_creation', methods = ['POST'])
def create_model():
    request_data = request.get_json()
    
    x_train = request_data['x_train']
    y_train = request_data['y_train']
    name = request_data['model_name']
    x_train = np.array(x_train)
    y_train = np.array(y_train) 
    model = model_creation(x_train, y_train)
    model.save(os.path.join(global_dir, name))
    response = {'model' : 'Creation Successful'}
    return jsonify(response)

@app.route('/API/load_model', methods = ['POST'])
def load_model():
    request_data = request.get_json()
    
    liste_action = request_data['option_action']
    name = request_data['model_name']
    liste_model = []
    
    for action in liste_action:
        liste_model.append(load_model(name))
        
    return jsonify(liste_model)
    
# @app.route('/API/model_compilation', methods = ['POST'])
# def compile_model():
#     request_data = request.get_json()
    
#     x_train = request_data['x_train']
#     y_train = request_data['y_train']
#     name_of_model = request_data['name_of_model']
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port = 5000, debug=True)