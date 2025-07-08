# train_model.py

import pandas as pd
import nltk
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Add labels
df_fake['label'] = 0
df_real['label'] = 1

# Combine and shuffle
df = pd.concat([df_fake, df_real], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    words = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_df=0.7, min_df=5, stop_words='english', ngram_range=(1,2))
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model and vectorizer
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(model, open("fake_news_model.pkl", "wb"))

print("✅ Model and vectorizer saved successfully.")
