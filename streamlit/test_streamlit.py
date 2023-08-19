# streamlit_app.py
import streamlit as st
import requests

# Définir l'URL de votre API FastAPI ou Flask
API_URL = "http://localhost:5000/api"  # Remplacez par l'URL de votre API

# Fonction pour appeler l'API FastAPI ou Flask et récupérer les données
def get_data_from_api(query):
    response = requests.get(f"{API_URL}/get_data", params={"query": query})
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Interface utilisateur Streamlit
st.title("Application avec Streamlit et API")

# Saisie de l'utilisateur
user_input = st.text_input("Entrez votre requête:")
if st.button("Rechercher"):
    if user_input:
        data = get_data_from_api(user_input)
        if data:
            st.write("Résultats:")
            st.write(data)
        else:
            st.write("Aucune donnée trouvée.")
            
            
