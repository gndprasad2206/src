import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd
import os
import nltk
import streamlit as st

def run_once():
    nltk.download("stopwords")

def preprocess_text(text):
    no_punctuation = [char for char in text if char not in string.punctuation]
    no_punctuation = "".join(no_punctuation)
    words = no_punctuation.lower().split()

    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    processed_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(processed_words)

from pathlib import Path
def get_absolute_path(relative_path):
    script_dir = Path(__file__).parent

    return script_dir / relative_path


def load_model():
    # MODEL_PATH = "../../model/resume_classifier.pkl"
    # MODEL_PATH = os.path.join("..", "..", "..", "model", "resume_classifier.pkl")
    MODEL_PATH = get_absolute_path("../../../model/resume_classifier.pkl")
    
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def load_vectorizer():
    # VECTORIZER_PATH = "../../model/vectorizer.pkl"
    # VECTORIZER_PATH = os.path.join("..", "..", "..", "model", "vectorizer.pkl")
    VECTORIZER_PATH = get_absolute_path("../../../model/vectorizer.pkl")


    with open(VECTORIZER_PATH, "rb") as file:
        return pickle.load(file)


class_labels = {
    0: "Match",
    1: "Moderate Match",
    2: "No adequate information",
    3: "Not Match",
}

get_class_key = lambda value: next((k for k, v in class_labels.items() if v == value), None)

def classify_message(message):
    preprocessed_message = preprocess_text(message)
    loaded_model = load_model()
    loaded_vectorizer = load_vectorizer()
    message_vector = loaded_vectorizer.transform([preprocessed_message])
    prediction = loaded_model.predict(message_vector)
    prediction_probs = loaded_model.predict_proba(message_vector)
    predicted_class_label = class_labels[prediction[0]]
    # print(f"Raw prediction output: {prediction[0]} ({predicted_class_label})")
    # print(f"Prediction probabilities: {prediction_probs[0]}")
    return predicted_class_label, prediction_probs[0]


