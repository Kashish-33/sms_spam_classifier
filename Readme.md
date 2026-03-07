# 📩 SMS Spam Classifier

A machine learning web application that classifies SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and a Voting Classifier ensemble model.

---

## 📌 Project Description

This project builds an end-to-end SMS spam detection system. It takes a raw SMS message as input, preprocesses it, converts it into numerical features using TF-IDF vectorization, and predicts whether the message is spam or legitimate using a trained machine learning model. The model is served via a Flask web application.

---

## 🧠 Model Details

### Algorithms Used
The model uses a **Voting Classifier** (soft voting) combining three algorithms:

| Model | Description |
|-------|-------------|
| **SVC** (Support Vector Classifier) | Uses a sigmoid kernel with probability estimates enabled |
| **MultinomialNB** (Multinomial Naive Bayes) | Well-suited for text classification with word frequency features |
| **ExtraTreesClassifier** | An ensemble of randomized decision trees for robust predictions |

### Text Preprocessing
Before training, each message goes through the following steps:
- Lowercasing
- Tokenization
- Removal of stopwords and punctuation
- Stemming using NLTK's PorterStemmer

### Vectorization
- **TF-IDF Vectorizer** is used to convert preprocessed text into numerical feature vectors

### Training
- Dataset: `spam.csv` (SMS Spam Collection dataset)
- Train/Test Split: 80/20
- Random State: 2

### Saved Files
- `model.pkl` — Trained MultinomialNB model
- `vectorizer.pkl` — Fitted TF-IDF vectorizer

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- NLTK
- Flask
- Pandas / NumPy
- Pickle

---

## 🚀 How to Run

1. Clone the repository:
   ```
   git clone https://github.com/Kashish-33/sms_spam_classifier.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```
   python flask_app.py
   ```

4. Open your browser and go to `http://127.0.0.1:5000`

---

## 📁 Project Structure

```
sms_spam_classifier/
├── flask_app.py          # Flask web application
├── model.pkl             # Trained model
├── vectorizer.pkl        # TF-IDF vectorizer
├── spam.csv              # Dataset
├── requirements.txt      # Dependencies
└── templates/            # HTML templates
```
