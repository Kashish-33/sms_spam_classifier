from flask import Flask, render_template, request
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()

# Text preprocessing function (same as training)
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # preprocess
    transformed = transform_text(message)

    # vectorize
    vector_input = vectorizer.transform([transformed])

    # predict
    prediction = model.predict(vector_input)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)