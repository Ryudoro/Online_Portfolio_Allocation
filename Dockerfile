# Utilisez une image de base appropriée
FROM python:3.8-slim

# Copiez les fichiers du projet dans le conteneur

WORKDIR /app

COPY ./Model /app/Model
COPY ./airflow_run.py /app/airflow_run.py
COPY ./requirements.txt /app/requirements.txt
COPY ./airflow.cfg /app/airflow2.cfg

# Installez les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# # Exposez le port pour l'API Flask
# EXPOSE 5000


# # Exécutez Airflow
# RUN airflow db init && \
#     airflow users create -f admin -l admin -p admin -u admin -r Admin -e admin@example.com && \
#     airflow webserver -p 8080 -H 0.0.0.0 -D && \
#     airflow scheduler

#CMD ["airflow", "scheduler"]

RUN cp airflow2.cfg airflow.cfg
RUN airflow db init
RUN airflow users create -f admin -l admin -p admin -u admin -e treymermier@gmx.fr -r Admin

