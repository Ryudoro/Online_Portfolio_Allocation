# streamlit_app.py
import streamlit as st
import requests
import matplotlib.pyplot as plt
import sys
import os
import datetime as dt


global_dir = '/home/project/Documents/Online_Portfolio_Allocation'
sys.path.append(global_dir)

# Définir l'URL de votre API FastAPI ou Flask
API_URL = "http://localhost:5000/api"  # Remplacez par l'URL de votre API

option_action = ("ALO.PA", "GOOGL")

if 'mode' not in locals():
    mode = "stastistical"
    
import joblib

def save_model(model, model_name):
    filename = f"{model_name}.joblib"
    joblib.dump(model, filename)
    
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
    model_list = []
    for action in option_action:
        model_list.append(0)
   
load_models()
st.pyplot(fig)
