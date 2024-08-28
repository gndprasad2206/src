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

nltk.download("stopwords")

def preprocess_text(text):
    no_punctuation = [char for char in text if char not in string.punctuation]
    no_punctuation = "".join(no_punctuation)
    words = no_punctuation.lower().split()

    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    processed_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(processed_words)



MODEL_PATH = "../../model/resume_classifier.pkl"
VECTORIZER_PATH = "../../model/vectorizer.pkl"

# Reload the model and vectorizer to verify
with open(MODEL_PATH, "rb") as file:
    loaded_model = pickle.load(file)

with open(VECTORIZER_PATH, "rb") as file:
    loaded_vectorizer = pickle.load(file)

class_labels = {
    0: "Match",
    1: "Moderate Match",
    2: "No adequate information",
    3: "Not Match",
}

def classify_message(message):
    preprocessed_message = preprocess_text(message)
    message_vector = loaded_vectorizer.transform([preprocessed_message])
    prediction = loaded_model.predict(message_vector)
    prediction_probs = loaded_model.predict_proba(message_vector)
    predicted_class_label = class_labels[prediction[0]]
    print(f"Raw prediction output: {prediction[0]} ({predicted_class_label})")
    print(f"Prediction probabilities: {prediction_probs[0]}")
    return predicted_class_label, prediction_probs[0]


def streamlit_main():
    st.write("### Enter a message to classify: ")
    message = "... this is a resume ... supposed to receive not enough information ..."
    result, probs = classify_message(message)

    st.write(f"The message is classified as: {result}")
    st.write("Class probabilities:")
    for index, prob in enumerate(probs):
        st.write(f"{class_labels[index]}: {prob:.4f}")


def main():
    # message = input("Enter a message to classify: ")
    message = "... this is a resume ... supposed to receive not enough information ..."
    result, probs = classify_message(message)

    print(f"The message is classified as: {result}")
    print("Class probabilities:")
    for index, prob in enumerate(probs):
        print(f"{class_labels[index]}: {prob:.4f}")

if __name__ == "__main__":
    # main()
    streamlit_main()
