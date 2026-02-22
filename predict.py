import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100

# Load saved model and tokenizer
model = load_model("model/sentiment_model.keras")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

NEUTRAL_THRESHOLD = 0.50  # If max confidence < 50%, classify as neutral

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
    return f"{pred_label} (confidence: {confidence}%)"


tests = {
    "POSITIVE": [
        "This movie was absolutely fantastic!",
        "I loved every moment of it, truly a masterpiece.",
        "Outstanding performance by the entire cast!",
        "Best purchase I have ever made, highly recommend it.",
        "The story was heartwarming and beautifully written.",
    ],
    "NEGATIVE": [
        "Worst experience ever.",
        "Terrible movie, complete waste of time and money.",
        "I hated it, the acting was awful and the plot made no sense.",
        "Extremely disappointed, would not recommend to anyone.",
        "The product broke after one day. Total garbage.",
    ],
    "NEUTRAL": [
        "The package was received and the contents were correct.",
        "The box arrived sealed and the product was inside.",
        "The box was received and the contents matched the order.",
        "The package arrived and everything inside was correct.",
        "The product was inside the sealed box when it arrived.",
    ],
}

for sentiment, messages in tests.items():
    print(f"\n--- Expected: {sentiment} ---")
    for msg in messages:
        result = predict_sentiment(msg)
        print(f"  [{result}] => {msg}")
