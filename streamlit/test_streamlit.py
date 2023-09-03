# streamlit_app.py
import streamlit as st
import requests
import matplotlib.pyplot as plt
import sys
import os
import datetime as dt
from tensorflow.keras.models import model_from_json

global_dir = '/home/project/Documents/Online_Portfolio_Allocation'
sys.path.append(global_dir)

# Définir l'URL de votre API FastAPI ou Flask
API_URL = "http://0.0.0.0:5000/api"  # Remplacez par l'URL de votre API

# option_action = ("ALO.PA", "GOOGL")
option_action = ["ALO.PA"]
if 'mode' not in locals():
    mode = "stastistical"
    
days_for_training = 100
days_for_testing = 0

# def save_model(model, model_name):
#     filename = f"{model_name}.joblib"
#     joblib.dump(model, filename)
    
# Fonction pour appeler l'API FastAPI ou Flask et récupérer les données
def get_data_from_api(stock_symbol):
    response = requests.get(f"{API_URL}/get_stock_data", params={"stock_symbol": stock_symbol})
    if response.status_code == 200:
        return response.json()
    else:
        return None

def transform_data_from_api(data_to_use,  days_for_training, days_for_testing, name_of_compagny):
    days_for_training = str(days_for_training)
    days_for_testing = str(days_for_testing)
    response = requests.post(f"{API_URL}/prepare_data", json={
    "data_to_use": data_to_use,
    "days_for_training": days_for_training,
    "days_for_testing": days_for_testing,
    "name_of_compagny": name_of_compagny
    })
    if response.status_code == 200:
        return response.json()
    else:
        return None
    

def create_model_compile(x_train, y_train):
    response = requests.get(f"{API_URL}/model_creation", params={"x_train": x_train, "y_train": y_train})
    if response.status_code == 200:
        return response.json()
    else:
        return None

  
def load_prediction_model(stock_symbol_list):
    response = requests.post(f"{API_URL}/load_model", json={"option_action": stock_symbol_list})
    if response.status_code == 200:
        return response.json()
    else:
        return None
     
     
def prediction_futur(days_in_futur, model, last_days_for_input, days_for_training, name_of_compagny):
    response = requests.post(f"{API_URL}/predict_futur", json={"days_in_futur": days_in_futur, "model": model, "last_days_for_input": last_days_for_input, "days_for_training": days_for_training, "name_of_compagny": name_of_compagny})
    if response.status_code == 200:
        return response.json()
    else:
        return None
      
# Interface utilisateur Streamlit
st.title("Online Portfolio Application")

st.write("""Bienvenu sur notre site, n'hesiter pas a utiliser ce site pour vos besoin en finance.
         Nous prevenons cependant qu'il est imprudent de se baser sur une simple machine pour 
         vendre ou acheter une action. N'hesitez pas a vous tenir au courant de l'actualite autour 
         d'une entreprise avant d'en acheter ses actions.""")

# Saisie de l'utilisateur
user_input = st.text_input("Entrez votre requête:")
if st.button("Rechercher"):
    if user_input:
        data = get_data_from_api(user_input)
        if data:
            data_to_use = data['data_to_use']
            st.write("Résultats:")
            fig, ax = plt.subplots(figsize = (10,8))
            data_plot = ax.plot(data_to_use)
            st.pyplot(fig)
        else:
            st.write("Aucune donnée trouvée.")
            
fig, ax = plt.subplots(figsize = (10,8))

with st.sidebar:
    
    st.write("Name of the compagny")
     
    name_of_compagny = st.selectbox("Selectionner une action:",
                                     options=option_action,
                                     index = 0,
                                     key = 'box_type')
    mode = st.selectbox("Quel mode d'utilisation voulez-vous utiliser:",
                                     options=("statistical", "prediction"),
                                     index = 0,
                                     key = 'mode')
    
    if mode == "prediction":
        days_of_predictions = st.slider("How much days in advance do you want to know the prediction", 0, 100, 7, 1)
        
        
    if mode == "statistical":
        stats = st.selectbox("Que souhaitez-vous utiliser ?:",
                                     options=("mean", "bollinger"),
                                     index = 0,
                                     key = 'statistical')
    if st.button("show graph"):
        data = get_data_from_api(name_of_compagny)
        
        if data:
            data_to_use = data['data_to_use']
            data_all = data['stock_data']
            Date = data_all['Date']
            data_plot = ax.plot(Date.values(), data_to_use)
            
            
    
    
# @st.cache_resource()
# def create_model():
#     for i in option_action:
#         days_for_training = 500
#         days_for_testing = 0
#         name_of_model = 'trained_model_'+i.replace('.','')+'.h5'
#         data = get_data_from_api(i)
#         data_to_use = data['data_to_use']
#         informations = transform_data_from_api(data_to_use, days_for_training, days_for_testing, name_of_compagny)
    
#         model_creation = create_model_compile(informations['x_train'], informations['y_train'])
  
# today = dt.datetime.now().date()

# filetime = dt.datetime.fromtimestamp(
# os.path.getctime('trained_model_ALOPA.h5'))
# if filetime.date() != today:
#     create_model()

@st.cache_resource()
def load_models():

    days_for_training = 100
    days_for_testing = 0

    data_list = []
    data_to_use_list = []
    X_train_list = []
    y_train_list = []
    for action in option_action:

        response_get_data_from_api = get_data_from_api(action)
        
        _ = response_get_data_from_api['data_to_use']
        data_list.append(response_get_data_from_api['stock_data'])
        data_to_use_list.append(_)
        response_data_transform = transform_data_from_api(_,  days_for_training, days_for_testing, action)
        
        X_train_list.append(response_data_transform['x_train'])
        y_train_list.append(response_data_transform['y_train'])
        
    model_list = load_prediction_model(option_action)
    
    return model_list, data_list, data_to_use_list, X_train_list, y_train_list
   
model_list,data_list, data_to_use_list, X_train_list, y_train_list = load_models()


if name_of_compagny == 'ALO.PA':
    model = model_from_json(model_list[0])
    model.summary()
    if mode == 'prediction':
        futur_prediction = prediction_futur(days_of_predictions, model_list[0], X_train_list[0][-1], days_for_training, name_of_compagny)
    
        st.write(futur_prediction)
st.pyplot(fig)
