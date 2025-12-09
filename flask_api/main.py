from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pickle.load(open("models/lr.pkl", "rb"))  # example model

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        data["fLength"], data["fWidth"], data["fSize"], data["fConc"],
        data["fConc1"], data["fAsym"], data["fM3Long"], data["fM3Trans"],
        data["fAlpha"], data["fDist"]
    ]])

    prediction = int(model.predict(features)[0])

    return jsonify({"prediction": prediction})


app.run(host="0.0.0.0", port=5000)
