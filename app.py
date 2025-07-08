import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load vectorizer and models
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

models = {
    "Logistic Regression": pickle.load(open("logistic_regression_model.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
    "XGBoost": pickle.load(open("xgboost_model.pkl", "rb"))
}

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App (Advanced)")

# --- Sidebar ---
st.sidebar.title("🧠 Choose Model")
model_choice = st.sidebar.radio("Select Classifier", list(models.keys()))
model = models[model_choice]

st.sidebar.markdown("💬 Example:")
if st.sidebar.button("Paste Example"):
    st.session_state["user_input"] = "NASA confirms discovery of water on Mars, opening door for future missions."

# --- Input Section ---
st.subheader("Enter News Text Below:")
user_input = st.text_area("Text to classify", key="user_input", height=200)

# --- Predict ---
if st.button("🔍 Predict"):
    if user_input.strip():
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        proba = model.predict_proba(transformed)[0]

        label = "🟢 REAL News" if prediction == 1 else "🔴 FAKE News"
        st.subheader(f"Prediction: {label}")

        # --- Confidence Bar ---
        st.markdown("### 🎯 Confidence")
        fig, ax = plt.subplots()
        ax.bar(["FAKE", "REAL"], proba, color=["red", "green"])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        st.markdown(f"**Selected Model:** `{model_choice}`")
    else:
        st.warning("Please enter text to classify.")
