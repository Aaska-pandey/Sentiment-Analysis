from fastapi import FastAPI
from pydantic import BaseModel

import joblib as JB
import re

import pandas as pd
from nltk.corpus import stopwords
import nltk


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# FastAPI app
app = FastAPI(title="Sentiment Analysis API", description="Predict sentiment of text reviews", version="1.0")

# Input schema
class ReviewRequest(BaseModel):
    review: str



# Define the preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Prediction function
def predict_sentiment(review_text):
    # Load saved model and vectorizer
    model = JB.load('sentiment_model.pkl')
    vectorizer = JB.load('tfidf_vectorizer.pkl')

    # Preprocess the input text
    review_clean = preprocess(review_text)

    # Transform text using the loaded vectorizer
    review_vector = vectorizer.transform([review_clean])

    # Predict
    prediction = model.predict(review_vector)[0]

    # If using Regressor: threshold at 0.5
    label = "positive" if prediction >= 0.5 else "negative"
    return label

# Example usage
sample_review = "bakwas tha bilkul"
print("Predicted Sentiment:", predict_sentiment(sample_review))