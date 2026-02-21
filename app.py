import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

MAX_LEN = 100
NEUTRAL_THRESHOLD = 0.50

model = load_model("model/sentiment_model.keras")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded, verbose=0)

    classes = ["negative", "neutral", "positive"]
    max_confidence = float(np.max(prediction))

    if max_confidence < NEUTRAL_THRESHOLD:
        pred_label = "neutral"
    else:
        pred_label = classes[np.argmax(prediction)]

    confidence = round(max_confidence * 100, 2)
    return pred_label, confidence


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("review", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment, confidence = predict_sentiment(text)
    return jsonify({"sentiment": sentiment, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)
