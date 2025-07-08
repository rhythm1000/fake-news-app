# app.py

import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# App title
st.title("📰 Fake News Detection App")
st.markdown("Detect whether a news article is **Real** or **Fake** using a Machine Learning model trained on TF-IDF features.")

# Load model and vectorizer
model_path = "fake_news_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

# Check file existence
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model files not found. Please run `train_model.py` to generate them.")
    st.stop()

# Load the model and vectorizer
model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Text input
user_input = st.text_area("📝 Enter the news content below:", height=250)

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        # Preprocess and transform
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        # Display result
        if prediction == 1:
            st.success("✅ This news article is **REAL**.")
        else:
            st.error("❌ This news article is **FAKE**.")

