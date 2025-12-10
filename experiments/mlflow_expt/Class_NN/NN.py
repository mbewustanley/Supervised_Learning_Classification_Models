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
#import dagshub
import logging

"""# dagshub expt intialization
dagshub.init(repo_owner='mbewustanley', repo_name='Supervised_Learning_Classification_Models', mlflow=True)"""


def get_logger(log_file="Neural_net_training.log"):
    # Ensure logs directory exists
    os.makedirs("data/logs", exist_ok=True)

    logger = logging.getLogger("Neural Net Training")
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Log to file
    file_handler = logging.FileHandler(f"data/logs/{log_file}")
    file_handler.setLevel(logging.DEBUG)

    # Log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = get_logger()




def load_data(path="data/split_data"):
    try:
        #import data
        X_train = np.load(f"{path}/X_train.npy")
        y_train = np.load(f"{path}/y_train.npy")
        X_valid = np.load(f"{path}/X_valid.npy")
        y_valid = np.load(f"{path}/y_valid.npy")

        logger.info("Successfully loaded training and validation data.")
        return X_train, y_train, X_valid, y_valid

    except FileNotFoundError as e:
        logger.error(f"Data files not found in {path}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise



def evaluate_neural_net(X_train, y_train, X_valid, y_valid):

    # Set MLflow tracking BEFORE training starts
    remote_server_url = "http://ec2-13-247-105-61.af-south-1.compute.amazonaws.com:5000"
    mlflow.set_tracking_uri(remote_server_url)
    mlflow.set_experiment("SLC-Neural-Net-Run")


    least_val_loss = float('inf')
    least_val_model = None
    epochs = 100
    try:
        logger.imfo("Neural Net mlflow run started"
                    )
        # Start MLflow run
        with mlflow.start_run() as run:

            for num_nodes in [16, 32, 64]:
                for dropout_prob in [0, 0.2]:
                    for lr in [0.01, 0.005, 0.001]:
                        for batch_size in [32, 64, 128]:

                            # Train the model
                            model, history = train_model(
                                X_train, y_train,
                                num_nodes=num_nodes,
                                dropout_prob=dropout_prob,
                                lr=lr,
                                batch_size=batch_size,
                                epochs=epochs
                            )

                            # Plot training history
                            plot_history(history)

                            # Evaluate on validation set
                            val_loss = model.evaluate(X_valid, y_valid)[0]

                            # Track best model
                            if val_loss < least_val_loss:
                                least_val_loss = val_loss
                                least_val_model = model

                            # Log params
                            mlflow.log_param('epochs', epochs)
                            mlflow.log_param('nodes', num_nodes)
                            mlflow.log_param('dropout', dropout_prob)
                            mlflow.log_param('learning_rate', lr)
                            mlflow.log_param('batch_size', batch_size)

                            # Log metrics
                            mlflow.log_metric('train_loss', history.history['loss'][-1])
                            mlflow.log_metric('val_loss', history.history['val_loss'][-1])
                            mlflow.log_metric('train_acc', history.history['accuracy'][-1])
                            mlflow.log_metric('val_acc', history.history['val_accuracy'][-1])

                            # Log model after each run
                            mlflow.tensorflow.log_model(
                                    model,
                                    "model",
                                    
                                )
            mlflow.tensorflow.load_model(least_val_model, "Classification NN", registered_model_name="ClassNeuraLNetMAGIC")
    except Exception as e:
        logger.error(f"neural net mlflow run failed")

    return least_val_model

                   

def main():
    try:
        X_train, y_train, X_valid, y_valid = load_data()
        initialize_neural_net(X_train, y_train, X_valid, y_valid)

    except Exception as e:
        logger.error(f"Neural net training failed: {e}")


if __name__ == "__main__":
    main()