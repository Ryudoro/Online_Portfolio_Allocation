from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.utils.dates import days_ago
import joblib
import sys
global_dir = '/home/project/Documents/Online_Portfolio_Allocation'
sys.path.append(global_dir)

from Model.input_creation import search_input
from Model.model_data_creation import model_data_creation
from Model.model_creation import model_creation
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import yfinance as yf

class CreateModelOperator(BaseOperator):

    @apply_defaults
    def __init__(
        self,
        stock_symbols,
        *args, **kwargs
        
        
    ):
        super().__init__(*args, **kwargs)
        self.stock_symbols = stock_symbols
        #self.name_of_model = 'trained_model_'+name_of_compagny.replace('.','')+'.h5'
        
    def execute(self, context):
        
        for stock_symbol in self.stock_symbols:
            ti = context['ti']
            x_train = ti.xcom_pull(task_ids='model_data_task', key=f'x_train_{stock_symbol}')
            y_train = ti.xcom_pull(task_ids='model_data_task', key=f'y_train_{stock_symbol}')
            
            model = model_creation(np.array(x_train), np.array(y_train))
            
            ti.xcom_push(f'model_{stock_symbol}', model.to_json())
            model.save('trained_model_'+stock_symbol.replace('.','')+'.h5')
        
class StoreDataInXComOperator(BaseOperator):

    @apply_defaults
    def __init__(
        self,
        task_id,
        data,
        key,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.task_id = task_id
        self.data = data
        self.key = key

    def execute(self, context):
        ti = context['ti']
        ti.xcom_push(self.key, self.data)
        
class DataCreation(BaseOperator):

    @apply_defaults
    def __init__(
        self,
        stock_symbols,
        *args, **kwargs
    ):
        
        #self.name_of_model = 'scaler'+name_of_compagny.replace('.','')+'.pkl'
        super().__init__(*args, **kwargs)
        self.stock_symbols = stock_symbols
        
    def execute(self, context):
        for stock_symbol in self.stock_symbols:
            ti = context['ti']
            data_to_use = ti.xcom_pull(task_ids='load_data_task', key=f'data_to_use_{stock_symbol}')
            x_train, y_train, X_test, scaler = model_data_creation(np.array(data_to_use))
            
            ti.xcom_push(f'x_train_{stock_symbol}', x_train.tolist())
            ti.xcom_push(f'y_train_{stock_symbol}', y_train.tolist())
            joblib.dump(scaler, 'scaler'+stock_symbol.replace('.','')+'.pkl')
        
class LoadDataOperator(BaseOperator):

    @apply_defaults
    def __init__(
        self,
        stock_symbols,
        period,
        jenkins,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.stock_symbols = stock_symbols
        self.period = period
        self.jenkins = jenkins

    def execute(self, context):
        
        for stock_symbol in self.stock_symbols:
            if self.jenkins:
                data = pd.read_csv('ALO.csv')
            else:
                data = yf.download(stock_symbol, period=self.period)
            
            target_column = 'Close'
            data_to_use = data[target_column].values
            data_to_use = data_to_use.tolist()
            ti = context['ti']
            ti.xcom_push(f'data_{stock_symbol}', data)
            ti.xcom_push(f'data_to_use_{stock_symbol}', data_to_use)
      
stock_symbols = ['ALO.PA', 'GOOGL']
period = '5y'
jenkins = False 
      
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

dag = DAG(
    'my_data_processing_dag',
    default_args=default_args,
    schedule_interval='@daily',  # ou votre fréquence
    catchup=False
)

load_data_task = LoadDataOperator(
    task_id='load_data_task',
    stock_symbols=stock_symbols,
    period=period,
    jenkins=jenkins,
    dag=dag,
)

def process_data(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='load_data_task', key='data')
    data_to_use = ti.xcom_pull(task_ids='load_data_task', key='data_to_use')
    
    return data,data_to_use

process_data_task = PythonOperator(
    task_id='process_data_task',
    python_callable=process_data,
    provide_context=True,
    dag=dag,
)


model_data_task = DataCreation(
    task_id='model_data_task',
    stock_symbols=stock_symbols,
    dag=dag,
)

create_model_task = CreateModelOperator(
    task_id='create_model_task',
    stock_symbols=stock_symbols,
    dag=dag,
)


load_data_task >> process_data_task
process_data_task >> model_data_task
model_data_task >> create_model_task