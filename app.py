import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🎭 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below and let the AI decide if it's *Positive* or *Negative*!")

# User input
user_input = st.text_area("📝 Your Review", height=200)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        # Preprocess and transform input
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        confidence = np.max(model.predict_proba(transformed_input)) * 100

        # Show results
        if prediction == 1:
            st.success(f"✅ Positive Review (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"❌ Negative Review (Confidence: {confidence:.2f}%)")