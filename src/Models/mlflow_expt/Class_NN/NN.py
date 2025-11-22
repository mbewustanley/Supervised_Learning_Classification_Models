import os
import warnings
import sys 


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from params import train_model, plot_history, eval_metrics

from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.tensorflow
import dagshub
import logging

# dagshub expt intialization
dagshub.init(repo_owner='mbewustanley', repo_name='Supervised_Learning_Classification_Models', mlflow=True)




#import data
X_train = np.load('split_data/X_train.npy')
y_train = np.load('split_data/y_train.npy')
X_valid = np.load('split_data/X_valid.npy')
y_valid = np.load('split_data/y_valid.npy')



with mlflow.start_run():
  
  #set defaults
  least_val_loss = float('inf')
  least_loss_model = None
  epochs=100
  for num_nodes in [16, 32, 64]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                model, history = train_model(X_train, y_train, num_nodes, 
                                            dropout_prob,
                                            lr, batch_size, epochs)
                plot_history(history)
                val_loss = model.evaluate(X_valid, y_valid)
                v_loss = val_loss[0]
                if v_loss < least_val_loss:
                    least_val_loss = v_loss
                    least_val_model = model
                mlflow.log_param('parameter name', 'value')
                mlflow.log_metric('metric name', 1)
                
                remote_server_url = "https://dagshub.com/mbewustanley/Supervised_Learning_Classification_Models.mlflow"
                mlflow.set_tracking_uri(remote_server_url)

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                #model registory does not work with file store
                if tracking_url_type_store != 'file':
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(
                        lr, "model", registered_model_name="ElasticnetWineModel")
                else:
                    mlflow.sklearn.log_model(lr, "model")
                   