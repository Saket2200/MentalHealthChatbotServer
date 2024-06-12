from flask import Flask, request, jsonify
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the FAQ data from the CSV file
df = pd.read_csv('/mnt/data/Mental_Health_FAQ.csv')

# Ensure the columns are correct
if 'question_id' in df.columns and 'Questions' in df.columns and 'Answers' in df.columns:
    questions = df['Questions'].values
    answers = df['Answers'].values
else:
    raise Exception("CSV file does not have the required columns: 'question_id', 'Questions', 'Answers'")

# Preprocess text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

# Preprocess all questions
preprocessed_questions = [preprocess(question) for question in questions]

# Load pre-trained BERT model and tokenizer for intent recognition
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    # Preprocess the user message
    preprocessed_message = preprocess(user_message)
    
    # Tokenize the user message
    inputs = tokenizer(preprocessed_message, return_tensors='pt', truncation=True, padding=True)
    
    # Get model predictions
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).item()
    
    # Find the index of the most similar question based on predictions
    best_match_index = predictions
    
    # Return the corresponding answer
    best_answer = answers[best_match_index]
    return jsonify({'response': f"Bot: {best_answer}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
