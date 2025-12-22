from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sales_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model not found at {MODEL_PATH}")
    model = None


# ✅ HOME ROUTE (FIXES 404)
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Sales Forecasting API is running successfully 🚀"
    })


@app.route("/predict_sales", methods=["POST"])
def predict_sales():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json

    try:
        features = np.array([[ 
            data["lag_1"],
            data["lag_7"],
            data["rolling_mean_7"],
            data["Price"],
            data["Discount"],
            data["Inventory Level"],
            data["Holiday/Promotion"],
            data["Competitor Pricing"]
        ]])

        prediction = model.predict(features)

        return jsonify({
            "predicted_units_sold": float(prediction[0])
        })

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
