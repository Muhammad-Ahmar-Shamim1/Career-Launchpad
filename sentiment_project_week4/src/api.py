import os
import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Set paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "bert_sentiment_model")

try:
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None
    device = None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or tokenizer is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.json
        if "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        sentiment = "Positive" if torch.argmax(probs) == 1 else "Negative"
        confidence = float(torch.max(probs))

        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Model API is running", "device": str(device)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
