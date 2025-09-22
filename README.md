
# Movie Review Sentiment Analyzer

This project is a web-based application built with Streamlit to analyze the sentiment of movie reviews, classifying them as Positive or Negative with a confidence score. It uses a Logistic Regression model trained on the IMDB movie review dataset, leveraging TF-IDF vectorization for text processing. The project demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, and deployment via a user-friendly web interface.

Features
Text Preprocessing: Converts text to lowercase, removes punctuation, and eliminates stopwords using NLTK.
Feature Extraction: Applies TF-IDF vectorization (limited to 5000 features) to transform text into numerical features.
Model: Logistic Regression classifier trained on the IMDB dataset (50,000 reviews).
Web Interface: Streamlit app for inputting reviews and viewing sentiment predictions with confidence scores.
Model Persistence: Saves and loads the trained model and vectorizer using Joblib for efficient deployment.

Technologies Used
Python 3.x
Libraries: Streamlit, Pandas, NumPy, Scikit-learn, NLTK, Joblib
Dataset: IMDB Movie Review Dataset (not included; download separately for training)
