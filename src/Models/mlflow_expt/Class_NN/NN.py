import os
import warnings
import sys 


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from params import train_model, plot_history, eval_metrics

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.tensorflow
import dagshub
import logging



#import data
X_train = np.load('split_data/X_train.npy')
y_train = np.load('split_data/y_train.npy')
X_valid = np.load('split_data/X_valid.npy')
y_valid = np.load('split_data/y_valid.npy')


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