import os
import json
import logging
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import logging


def get_logger(log_file="training.log"):
    # Ensure logs directory exists
    os.makedirs("data/logs", exist_ok=True)

    logger = logging.getLogger("Model Training")
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
        X_train = np.load(f"{path}/X_train.npy")
        y_train = np.load(f"{path}/y_train.npy")

        logger.info("Successfully loaded training data.")
        return X_train, y_train

    except FileNotFoundError as e:
        logger.error(f"Data files not found in {path}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise



# initialize Knn model
def initialize_models():
    try:
        models = {
            "knn": KNeighborsClassifier(n_neighbors=1),
            "lr": LogisticRegression(),
            "nb": GaussianNB(),
            "svm": SVC()
        }

        logger.info("Models initialized successfully.")
        return models

    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise



def train_models(models: dict, X_train, y_train):
    trained = {}
    
    for name, model in models.items():
        try:
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            trained[name] = model
            logger.info(f"{name} training completed.")
        except Exception as e:
            logger.error(f"Training failed for {name}: {e}")
            raise

    return trained




def save_models(models: dict, output_dir="models"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        try:
            file_path = f"{output_dir}/{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"Saved {name} model to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save {name}: {e}")
            raise



def main():
    try:
        logger.info("Pipeline started.")

        # load
        X_train, y_train = load_data()

        # init
        models = initialize_models()

        # train
        trained_models = train_models(models, X_train, y_train)

        # save
        save_models(trained_models)

        logger.info("Training completed successfully.")

    except Exception as e:
        logger.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()

