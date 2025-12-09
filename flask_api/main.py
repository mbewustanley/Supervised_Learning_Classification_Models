from flask import Flask, request, jsonify
import numpy as np
import pickle
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model configuration
with open("configs/model_config.json", "r") as f:
    MODEL_REGISTRY = json.load(f)

def load_model(model_key):
    """Loads model dynamically based on the user selection"""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_key}' is not registered")
    
    model_path = MODEL_REGISTRY[model_key]
    return pickle.load(open(model_path, "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    # Retrieve model selection
    model_key = data.get("model", None)
    if model_key is None:
        return jsonify({"error": "No model selected"}), 400

    # Load selected model
    try:
        model = load_model(model_key)
    except Exception as e:
        return jsonify({"error": f"Error loading model: {str(e)}"}), 500

    # Extract features
    try:
        features = np.array([[
            data["fLength"], data["fWidth"], data["fSize"], data["fConc"],
            data["fConc1"], data["fAsym"], data["fM3Long"], data["fM3Trans"],
            data["fAlpha"], data["fDist"]
        ]])
    except KeyError:
        return jsonify({"error": "Missing input parameters"}), 400

    # Predict
    try:
        pred = int(model.predict(features)[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"prediction": pred, "model_used": model_key})


app.run(host="0.0.0.0", port=5000)