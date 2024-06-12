# app.py

from flask import Flask, request, jsonify
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

nltk.download('punkt')

app = Flask(__name__)

# Load your CSV file into a DataFrame
df = pd.read_csv('Mental_Health_FAQ.csv')

# Ensure the CSV has columns 'questions' and 'answer'
texts = df['Questions'].values
answers = df['Answers'].values

# Train a simple model for intent classification
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words='english')
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, answers)

# Save the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump((vectorizer, model), model_file)

def get_response(message):
    with open('model.pkl', 'rb') as model_file:
        vectorizer, model = pickle.load(model_file)
    tokens = word_tokenize(message)
    X = vectorizer.transform([' '.join(tokens)])
    predicted_answer = model.predict(X)[0]
    response = f"The answer is: {predicted_answer}"
    return response

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Mental Health Chatbot API!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    response = get_response(message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
