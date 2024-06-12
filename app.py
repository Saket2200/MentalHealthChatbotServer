from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('Mental_Health_FAQ.csv')

# Initialize the TfidfVectorizer and fit it on the questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['questions'])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    user_vec = vectorizer.transform([user_message])
    similarity = cosine_similarity(user_vec, X)
    best_match_idx = similarity.argmax()
    response = df['answers'].iloc[best_match_idx]

    # Format the response (removing "The answer is: ")
    formatted_response = response.replace("The answer is: ", "")
    return jsonify({'response': formatted_response})

if __name__ == '__main__':
    app.run(debug=True)
