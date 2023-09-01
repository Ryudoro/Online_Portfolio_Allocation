# Utilisez une image de base appropriée
FROM python:3.8-slim

# Copiez les fichiers du projet dans le conteneur

WORKDIR /app

COPY ./Model /app/Model
COPY ./airflow_run.py /app/airflow_run.py
COPY ./requirements.txt /app/requirements.txt

# Installez les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port pour l'API Flask
EXPOSE 5000

# Exécutez Airflow
CMD ["airflow", "scheduler"]
