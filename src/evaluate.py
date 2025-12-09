import os
import json
import numpy as np
import pickle
import logging

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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


def load_test_data(path="split_data"):
    try:
        X_test = np.load(f"{path}/X_test.npy")
        y_test = np.load(f"{path}/y_test.npy")

        logger.info("Successfully loaded test data.")
        return X_test, y_test

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


def main():
    try:
        logger.info("Evaluation pipeline started.")

        # 1. load test data
        X_test, y_test = load_test_data()

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

    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")


if __name__ == "__main__":
    main()