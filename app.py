from flask import Flask, request, jsonify
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

nltk.download('punkt')

# Load your CSV file into a DataFrame
df = pd.read_csv('Mental_Health_FAQ.csv')

# Assume the CSV has columns 'Questions' and 'Answers'
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

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    with open('model.pkl', 'rb') as model_file:
        vectorizer, model = pickle.load(model_file)
    tokens = word_tokenize(message)
    X = vectorizer.transform([' '.join(tokens)])
    predicted_answer = model.predict(X)[0]
    response = {"response": predicted_answer}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
