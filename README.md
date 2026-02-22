<div align="center">
  
# 💬 Sentiment Analysis

### 🧠 ANN-Powered Sentiment Classifier built with Flask

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ANN_Model-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep_Learning-red?style=for-the-badge&logo=keras)
![Bootstrap](https://img.shields.io/badge/UI-Bootstrap-purple?style=for-the-badge&logo=bootstrap)

</p>
</div>

---

## ✨ Overview

**Sentiment Analysis** is a modern Flask web app that classifies user-entered text as **Positive**, **Negative**, or **Neutral** using an **Artificial Neural Network (ANN)**.

Instead of relying on rule-based keyword matching, the model learns contextual patterns from thousands of real reviews — delivering accurate, confidence-scored predictions in real time.

📦 From a simple **Jupyter Notebook experiment**, this project has been upgraded into a **fully interactive web application** with:

- ✔ Text Input
- ✔ Real-time Prediction
- ✔ Confidence Score
- ✔ Sentiment Label with Visual Feedback
- ✔ Clean Glassmorphism UI

All in seconds.

---

## 🎯 Demo Flow

```
Enter Review Text
      ↓
Tokenize + Pad Sequence
      ↓
ANN Model Inference
      ↓
Apply Confidence Threshold
      ↓
Display Sentiment + Confidence Score
```

---

## 📸 Screenshots

### 💻 Interface

![Interface](sampleScreenshots/Screenshot%20(1895).png)

### Result: Positive Sentiment

![Positive Result](sampleScreenshots/Screenshot%20(1896).png)

### Result: Negative Sentiment

![Negative Result](sampleScreenshots/Screenshot%20(1897).png)

### Result: Neutral Sentiment

![Neutral Result](sampleScreenshots/Screenshot%20(1898).png)

---

## 🔥 Features

### 🤖 Model & Prediction

* ANN with Embedding + GlobalAveragePooling layers
* 3-class softmax output: Negative / Neutral / Positive
* Tokenizer with OOV token for unseen words

### 📊 Smart Output

* Predicted sentiment label
* Confidence percentage
* Color-coded visual feedback (green / red / grey)
* Instant JSON response via REST API

### 💎 UI/UX

* Glassmorphism card design
* Animated gradient background
* Smooth result transitions
* Mobile responsive layout
* Poppins font + custom CSS styling

### ⚡ Backend

* Flask REST API (`/predict` endpoint)
* JSON request/response handling
* Pre-loaded model and tokenizer at startup
* Lightweight and fast inference

---

## 🧠 How It Works (Simple)

### Step 1 — Tokenize Input

```
Raw Text → Tokenizer → Integer Sequences
```

Example:

```
"This movie was great!" → [4, 56, 12, 87]
```

---

### Step 2 — Pad Sequence

```
Sequences → Fixed length (MAX_LEN = 100)
```

Shorter sequences are zero-padded; longer ones are truncated.

---

### Step 3 — ANN Inference

```
Padded Sequence → Embedding → GlobalAveragePooling → Dense → Softmax
```

Output:

```
[negative_prob, neutral_prob, positive_prob]
```

---

### Step 4 — Pick the Label

```
[negative_prob, neutral_prob, positive_prob]
```

The model selects the class with the **highest confidence score** as the final predicted sentiment.

---

## 🏗️ Tech Stack

| Layer            | Tech                          |
| ---------------- | ----------------------------- |
| Backend          | Flask                         |
| Deep Learning    | TensorFlow / Keras (ANN)      |
| Text Processing  | Keras Tokenizer + Pad Sequences |
| Math             | NumPy, Pandas                 |
| Frontend         | HTML + CSS + JavaScript       |
| Fonts / Icons    | Google Fonts (Poppins)        |

---

## ⚠️ Known Limitations

### 😐 Neutral Label Detection

The model currently struggles with accurately detecting **Neutral** sentiment. The primary reason is the **quality and balance of the training dataset** — neutral examples are often underrepresented or inconsistently labeled compared to clearly positive or negative reviews.

As a result, borderline texts may be misclassified as positive or negative instead of neutral.

> Improving the neutral class requires a cleaner, more diverse, and better-balanced dataset with well-defined neutral examples.

---

## 📂 Project Structure

```
Sentiment-Analysis-using-ANN/
│
├── app.py
├── predict.py
├── requirements.txt
│
├── data/
│   ├── sentiment_data.csv
│   ├── EcoPreprocessed.csv
│   └── train_df.csv
│
├── model/
│   ├── modal.py
│   ├── sentiment_model.keras
│   └── tokenizer.pkl
│
├── notebooks/
│   └── Sentiment-Analysis.ipynb
│
├── templates/
│   └── index.html
│
├── sampleScreenshots/
│
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone repo

```bash
git clone https://github.com/SACHIN-S-2004/Sentiment-Analysis-using-ANN.git
cd Sentiment-Analysis-using-ANN
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run app

```bash
python app.py
```

---

### 4️⃣ Open browser

```
http://127.0.0.1:5000
```

---

## 📈 Example Results

| Input Text                              | Prediction | Confidence |
| --------------------------------------- | ---------- | ---------- |
| "This movie was absolutely fantastic!"  | Positive   | 97.83%     |
| "Worst experience ever."                | Negative   | 95.12%     |
| "The package arrived and was correct."  | Neutral    | 83.60%     |
| "Truly a masterpiece, loved every bit!" | Positive   | 96.45%     |
| "Complete waste of time and money."     | Negative   | 94.70%     |

---

## 🎓 Learning Outcomes

This project demonstrates:

- ✔ Supervised Deep Learning (ANN)
- ✔ Natural Language Processing (NLP) fundamentals
- ✔ Text tokenization and sequence padding
- ✔ Flask backend development
- ✔ Practical ML model deployment

---

## ⭐ If you like this project

Give it a star — it helps a lot!
