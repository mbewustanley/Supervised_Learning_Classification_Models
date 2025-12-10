import os
import json
import numpy as np
import pickle
import logging
import mlflow
import mlflow.tensorflow
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from train import initialize_neural_net, plot_NN_history
from sklearn.metrics import accuracy_score, precision_score, recall_score


def get_logger(log_file="evaluation.log"):
    # Ensure logs directory exists
    os.makedirs("data/logs", exist_ok=True)

    logger = logging.getLogger("Model Evaluation")
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


def load_test_data(path="data/split_data"):
    try:
        X_test = np.load(f"{path}/X_test.npy")
        y_test = np.load(f"{path}/y_test.npy")
        X_train = np.load(f"{path}/X_train.npy")
        y_train = np.load(f"{path}/y_train.npy")
        X_valid = np.load(f"{path}/X_valid.npy")
        y_valid = np.load(f"{path}/y_valid.npy")

        logger.info("Successfully loaded test data.")
        return X_test, y_test, X_train, y_train, X_valid, y_valid

    except FileNotFoundError as e:
        logger.error(f"Test data files not found: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading test data: {e}")
        raise


def load_models(model_dir="models"):
    """Load all pickle models in the models directory."""
    models = {}
    try:
        for file in os.listdir(model_dir):
            if file.endswith(".pkl"):
                path = os.path.join(model_dir, file)
                name = file.replace(".pkl", "")

                with open(path, "rb") as f:
                    models[name] = pickle.load(f)

                logger.info(f"Loaded model: {name}")

        return models
    
    except FileNotFoundError:
        logger.error(f"Model directory not found: {model_dir}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading models: {e}")
        raise



def evaluate_model(model, X_test, y_test):
    """Compute accuracy, precision, recall for one model."""
    try:
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="macro", zero_division=0)
        recall = recall_score(y_test, preds, average="macro", zero_division=0)

        return accuracy, precision, recall

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise



def evaluate_all_models(models: dict, X_test, y_test, output_file="evaluation_results.json"):
    """Evaluate every loaded model and log the results."""
    results = {}

    for name, model in models.items():
        try:
            logger.info(f"Evaluating model: {name}")

            acc, prec, rec = evaluate_model(model, X_test, y_test)

            logger.info(f"{name} → Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

            results[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec
            }

        except Exception as e:
            logger.error(f"Failed evaluating {name}: {e}")

    # ---- SAVE RESULTS TO JSON ----
    try:
        os.makedirs("data/metrics", exist_ok=True)
        output_path = os.path.join("data/metrics", output_file)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"Saved evaluation results to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")
        raise

    return results


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
                            model, history = initialize_neural_net(
                                X_train, y_train,
                                num_nodes=num_nodes,
                                dropout_prob=dropout_prob,
                                lr=lr,
                                batch_size=batch_size,
                                epochs=epochs
                            )

                            # Plot training history
                            plot_NN_history(history)

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
           # mlflow.tensorflow.load_model(least_val_model, "Classification NN", registered_model_name="ClassNeuraLNetMAGIC")
    except Exception as e:
        logger.error(f"neural net mlflow run failed")

    return least_val_model




def main():
    try:
        logger.info("Evaluation pipeline started.")

        # 1. load test data
        X_test, y_test, X_train, y_train, X_valid, y_valid = load_test_data()

        # 2. load models
        models = load_models()

        # 3. evaluate models
        results = evaluate_all_models(models, X_test, y_test)

        logger.info("Evaluation complete.")

        # 4. (Optional) print results to console
        for name, metrics in results.items():
            logger.info(
                f"{name.upper()} → Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}"
            )

        # evaluate neural net
        least_val_model = evaluate_neural_net(X_train, y_train, X_valid, y_valid)

    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")


if __name__ == "__main__":
    main()