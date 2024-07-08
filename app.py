# Import necessary libraries
from flask import Flask, render_template, request, redirect
import os
import textract
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Define a list of users with their email and password
users = [
    {'email': 'h12325956@gmail.com', 'password': 'happy56_n'},
    {'email': 'user2@example.com', 'password': 'password2'},
    {'email': 'user3@example.com', 'password': 'password3'}
]

# Load the dataset
df = pd.read_csv(r"archive (9)\UpdatedResumeDataSet.csv")

# Tokenize and preprocess the dataset
stop_words = set(stopwords.words('english'))
df['Cleaned_Resume'] = df['Resume'].apply(lambda x: ' '.join([token.lower() for token in word_tokenize(x) if token.isalpha() and token.lower() not in stop_words]))

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit-transform the dataset
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Resume'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if the email and password match the stored credentials
        user = next((user for user in users if user['email'] == email and user['password'] == password), None)
        if user:
            return redirect('/upload')  # Redirect to the upload route
        else:  # Render the login page with an error message if login fails
            return render_template('index.html', error='Invalid email or password. Please try again.')

@app.route('/upload')
def upload():
    return render_template('upload_resume.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    # Get the uploaded files
    uploaded_files = request.files.getlist('file')

    # Iterate over each uploaded file
    parsed_data = []
    for file in uploaded_files:
        # Save the file to the uploads folder
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Extract text from the uploaded file using textract
        try:
            text = textract.process(file_path, encoding='utf-8')
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            continue

        # Tokenize and preprocess the uploaded resume
        tokens = word_tokenize(text.decode('utf-8'))
        parsed_text = ' '.join([token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words])
    
        # Append the parsed text to parsed_data
        parsed_data.append(parsed_text)
        
        # Print the parsed_data
        print(parsed_data)

    # Transform the uploaded resumes
    uploaded_tfidf_matrix = tfidf_vectorizer.transform(parsed_data)

    # Calculate cosine similarity between uploaded resumes and dataset resumes
    similarities = cosine_similarity(uploaded_tfidf_matrix, tfidf_matrix)

    # Get the unique job roles
    unique_jobs = df['Category'].unique()

    # Generate leaderboard data
    leaderboard_data = []
    for i, sim_scores in enumerate(similarities):
        job_rankings = []
        for job in unique_jobs:
            job_index = df[df["Category"] == job].index[0]
            ranking = round(sim_scores[job_index] * 100, 2)  # Compute ranking as a percentage
            job_rankings.append({'Job': job, 'Ranking': ranking})
        
        # Sort job rankings based on the ranking percentage in descending order
        job_rankings.sort(key=lambda x: float(x['Ranking']), reverse=True)
        
        leaderboard_data.append({'Resume': f'Resume {i+1}', 'Job Rankings': job_rankings})

    # Render the leaderboard template with parsed data and leaderboard data
    return render_template('leaderboard.html', parsed_data=parsed_data, leaderboard_data=leaderboard_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
