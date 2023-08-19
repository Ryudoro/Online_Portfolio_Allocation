# app.py
from flask import Flask, request, jsonify

app = Flask(__name__)

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)