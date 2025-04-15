from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("fraud_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = request.json.get("features")
        if not features or len(features) != 29:
            return jsonify({"error": "Provide exactly 29 features"}), 400

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        result = "Fraud" if prediction == 1 else "Legit"

        return jsonify({"prediction": int(prediction), "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
