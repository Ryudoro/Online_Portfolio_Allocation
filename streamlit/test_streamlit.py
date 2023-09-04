# streamlit_app.py
import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
import datetime as dt
from tensorflow.keras.models import model_from_json
import pandas as pd
import numpy as np
import mplfinance as fplt
st.set_option('deprecation.showPyplotGlobalUse', False)

global_dir = '/home/supersymmetry/Documents/Online_Portfolio_Allocation'
sys.path.append(global_dir)
from Stats.statistics import bollinger
from Stats.statistics import rolling_mean
# Définir l'URL de votre API FastAPI ou Flask
API_URL = "http://0.0.0.0:5000/api"  # Remplacez par l'URL de votre API

# option_action = ("ALO.PA", "GOOGL")
option_action = ["ALO.PA", "GOOGL"]
if 'mode' not in locals():
    mode = "stastistical"
    
action_dict = {"ALO.PA" : 0, "GOOGL": 1}
days_for_training = 500
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
# user_input = st.text_input("Entrez votre requête:")
# if st.button("Rechercher"):
#     if user_input:
#         data = get_data_from_api(user_input)
#         if data:
#             data_to_use = data['data_to_use']
#             st.write("Résultats:")
#             fig, ax = plt.subplots(figsize = (10,8))
#             data_plot = ax.plot(data_to_use)
#             st.pyplot(fig)
#         else:
#             st.write("Aucune donnée trouvée.")
            
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
    
    data_choice = st.selectbox("Période d'anaylse", options = ("7j", "1m", "6m", "1y", "5y"), index = 0, key = 'date')
    if mode == "prediction":
        days_of_predictions = st.slider("How much days in advance do you want to know the prediction", 0, 100, 7, 1)
        
        
    if mode == "statistical":
        stats = st.selectbox("Que souhaitez-vous utiliser ?:",
                                     options=("mean", "bollinger"),
                                     index = 0,
                                     key = 'statistical')
        show_base_graph = st.checkbox("Voulez vous voir le cours réel de l'action", value = True)
        
        if stats == 'mean':
            rolling_mean_value = st.slider("Sur combien de jours voulez vous effectuer la moyenne glissante?", 0,100,20,1)
        if stats == 'bollinger':
            rolling_bollinger_value = st.slider("Sur combien de jours voulez vous effectuer l'ecart quadratique moyen?", 0,100,20,1)
            
    stop_lossm = st.checkbox("Voulez vous ajouter une stop loss ? Cela peut vous aider pour fixer vos objectifs")
    
    if stop_lossm:
        stop_loss = st.slider("A quelle valeur souhaitez vous placer votre stop loss ?", 0, 100, 10, 1)
        stop_loss_incli = st.slider("quelle inclinaison ?", 0., 1., 0., 0.1)
    # if st.button("show graph"):
    #     data = get_data_from_api(name_of_compagny)
        
    #     if data:
    #         data_to_use = data['data_to_use']
    #         data_all = data['stock_data']
    #         Date = data_all['Date']
    #         data_plot = ax.plot(Date.values(), data_to_use)
            
            
    
    
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

    days_for_training = 500
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

plot_tab, result_tab, resume_tab = st.tabs(["Résultats", "Détails", "Project Info"])

with plot_tab:
    
    #if name_of_compagny == 'ALO.PA':
    model = model_from_json(model_list[action_dict[name_of_compagny]])
    model.summary()
    data_range = data_to_use_list[action_dict[name_of_compagny]]
    data = pd.DataFrame(data_list[action_dict[name_of_compagny]])
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    
    fig, ax = plt.subplots(figsize = (10,8))
    
    # Filtrez les données en fonction de la durée sélectionnée
    if data_choice == "7j":
        filtered_data = data[-7:]  # Dernière semaine
        fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    elif data_choice == "1m":
        fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        filtered_data = data[-30:]  # Dernier mois
    elif data_choice == "6m":
        fig.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        filtered_data = data[-180:]  # Derniers 6 mois
    elif data_choice == "1y":
        fig.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        filtered_data = data[-365:]  # Dernière année
    elif data_choice == "5y":
        fig.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        filtered_data = data  # 5 ans complets
    
    data_index = pd.to_datetime(filtered_data.index)

    
    # fig.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    # fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation = 45)
    if stop_lossm:
        def f(x):
            return stop_loss_incli * x + stop_loss
        if len(filtered_data) > 100:
                    indices = np.linspace(0,len(filtered_data)-1, 100, dtype=int)
                    filtered_data_2 = filtered_data.iloc[indices]
                    
        ax.plot(filtered_data_2.index, [f(x) for x in range(100)])
    if mode == 'prediction':
        futur_prediction = prediction_futur(days_of_predictions, model_list[action_dict[name_of_compagny]], X_train_list[action_dict[name_of_compagny]][-1], days_for_training, name_of_compagny)
        last_date = filtered_data.index[-1]
        date_index_future = pd.date_range(start=last_date, periods=days_of_predictions+1)

        futur_pred = np.array(list(futur_prediction.values())).reshape(-1)


        ax.plot(date_index_future, futur_pred , color='green', label='Prix prédit') 
        ax.plot(data_index, filtered_data['Close'], label = 'Cours de clôture')
        #st.write(futur_prediction['futur_prediction'])
        
    if mode == 'statistical':
        if stats == 'bollinger':
            
            if len(filtered_data['Close'])> 2* rolling_bollinger_value:
                
                boll_down, boll_up = bollinger(filtered_data['Close'], rolling_bollinger_value)
                
                if len(filtered_data) > len(boll_down):
                    indices = np.linspace(0,len(filtered_data)-1, len(boll_down), dtype=int)
                    filtered_data_2 = filtered_data.iloc[indices]
                ax.plot(pd.to_datetime(filtered_data_2.index), boll_down, label = 'Bollinger band down')
                ax.plot(pd.to_datetime(filtered_data_2.index), boll_up, label = 'Bollinger band up')
            else:
                st.write("Attention à choisir une moyenne glissante sur interval de temps plus court")
        if stats == 'mean':
            
            if len(filtered_data['Close']) > 2 * rolling_mean_value:
                mean = rolling_mean(filtered_data['Close'], rolling_mean_value)
                if len(filtered_data) > len(mean):
                    indices = np.linspace(0,len(filtered_data)-1, len(mean), dtype=int)
                    filtered_data_2 = filtered_data.iloc[indices]
                ax.plot(pd.to_datetime(filtered_data_2.index), mean)
            else:
                st.write("Attention à choisir une moyenne glissante sur un interval de temps plus court.")
                
        if show_base_graph:
            ax.plot(data_index, filtered_data['Close'], label = 'Cours de clôture')

    
    ax.set_xlabel('Date')
    ax.set_ylabel('Prix de clôture')
    ax.set_title(f"Evolution du cours de l'action {name_of_compagny}")
    ax.legend()
    ax.grid(True)
    
    fig2, ax2 = plt.subplots(figsize = (10,8))
    
    ax2.bar(data_index, filtered_data['Volume'], label ='Volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.set_title(f"Evolution du volume de l'action {name_of_compagny}")
    
    
    ax4 = fplt.plot(filtered_data, volume = True, ylabel = 'price', type = 'candle', style = 'charles')
        

    st.pyplot(fig)

    st.pyplot(fig2)

    # st.pyplot(fig3)

    st.pyplot(ax4)
    
with result_tab:
    st.write(data)
    
with resume_tab:
    st.write("Price prediction using tensorflow")