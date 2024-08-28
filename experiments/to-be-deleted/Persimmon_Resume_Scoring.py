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


# Define the path to your Excel file
# MODEL_DIR = os.path.join("persimmon-ai","prototypes","ml", "data")
# TEST_RESUME_JDs = os.path.join(MODEL_DIR, "Test_Resume_and_JDs.xlsx")
# print("Constructed file path:", TEST_RESUME_JDs)

# MODEL_DIR = os.path.join("persimmon-ai","prototypes","ml", "data")
# TEST_RESUME_JDs = os.path.join(MODEL_DIR, "Test_Resume_and_JDs.xlsx")
TEST_RESUME_JDs = "../../data/Test_Resume_and_JDs.xlsx"
print("Constructed file path:", TEST_RESUME_JDs)


data = pd.read_excel(TEST_RESUME_JDs)


data = data.dropna()
data["RD"] = data["Description"] + " " + data["Resume Text"]
data["RD"] = data["RD"].apply(preprocess_text)
data["Match Level"] = data["Match Level"].replace("No Match", "Not Match")

from imblearn.over_sampling import RandomOverSampler

X = data.drop(columns=["Match Level"])
y = data["Match Level"]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_resampled["Match Level"])

X_train, X_test, y_train, y_test = train_test_split(
    df_resampled["RD"], y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

import xgboost as xgb

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train_vec, y_train)

print(classification_report(y_train, model.predict(X_train_vec)))
print(classification_report(y_test, model.predict(X_test_vec)))

# Define the path to your pickle file
print("Current working directory2:", os.getcwd())

MODEL_DIR = os.path.join("ml", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "resume_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Ensure the directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Save the model and vectorizer
with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

with open(VECTORIZER_PATH, "wb") as file:
    pickle.dump(vectorizer, file)

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

message = input("Enter a message to classify: ")
result, probs = classify_message(message)

print(f"The message is classified as: {result}")
print("Class probabilities:")
for index, prob in enumerate(probs):
    print(f"{class_labels[index]}: {prob:.4f}")
