import os
import json
import logging
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


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



# initialize sklearn models
def initialize_sklearn_models(output_path="configs/model_config.json"):
    try:
        models = {
            "knn": KNeighborsClassifier(n_neighbors=1),
            "lr": LogisticRegression(),
            "nb": GaussianNB(),
            "svm": SVC()
        }

        configs = {  
            key: f"models/{key}.pkl "for key in models.keys()
            }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # write JSON file
        with open(output_path, "w") as f:
            json.dump(configs, f, indent=4)

        logger.info("Models initialized successfully.")
        return models

    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise



# to be exported into evaluate.py
def initialize_neural_net(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    try:
        nn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10)),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(num_nodes, activation='relu'),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        nn_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss='binary_crossentrophy',
            metrics=['accuracy']
        )

        history = nn_model.fit(X_train, y_train, 
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2,
                            verbose=0)
        
        logger.info("Neural Net Initialized and fit successfully")
        return nn_model, history
    
    except Exception as e:
        logger.error(f"Error initializing neural net: {e}")
        raise
    
#to be exported to evaluate.py
def plot_NN_history(history):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_ylabel('Binary_crossentrophy')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show()


def train_sklearn_models(models: dict, X_train, y_train,):  # sklearn model training
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



def save_sklearn_models(models: dict, output_dir="models"):
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
        models = initialize_sklearn_models()

        # train
        trained_models = train_sklearn_models(models, X_train, y_train)

        # save
        save_sklearn_models(trained_models)

        logger.info("Training completed successfully.")

    except Exception as e:
        logger.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()

