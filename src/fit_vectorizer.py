import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Starting vectorizer fitting...")

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

import os

# Load dataset
print("Loading dataset...")
dataset_path = os.path.join(os.path.dirname(__file__), "resume_dataset.csv")
df = pd.read_csv(dataset_path)

# Clean resumes
print("Cleaning resumes...")
df['resume_clean'] = df['resume'].apply(clean_text)

# TF-IDF Vectorizer
print("Fitting TfidfVectorizer with 800 max features...")
tfidf_vectorizer = TfidfVectorizer(max_features=800)
X_tfidf = tfidf_vectorizer.fit_transform(df['resume_clean'])

# Save vectorizer
print("Saving tfidf_vectorizer.pkl...")
vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
joblib.dump(tfidf_vectorizer, vectorizer_path)
print("Vectorizer successfully fit and saved to tfidf_vectorizer.pkl!")
