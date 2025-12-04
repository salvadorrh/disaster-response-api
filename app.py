# app.py
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_model(path):
    with open(path, "rb") as f:
        return joblib.load(f)
    
# Load vectorizer
vectorizer = load_model("models/final_vectorizer.pkl")

# Lables -> ['aid_related', 'water', 'food', 'shelter']
models = {
    "aid_related": load_model("models/SGDClassifier_tfidf_5000_0.pkl"),
    "water": load_model("models/SGDClassifier_tfidf_5000_1.pkl"),
    "food": load_model("models/SGDClassifier_tfidf_5000_2.pkl"),
    "shelter": load_model("models/SGDClassifier_tfidf_5000_3.pkl")
}

# Home route
@app.route("/", methods=["GET"])
def home():
    return {"status": "alive"}

# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    X = vectorizer.transform([text])

    # For each label
    results = {}
    for label, model in models.items():
        pred = int(model.predict(X)[0])
        results[label] = pred

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5000)