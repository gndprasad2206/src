# prototypes/ml/notebooks/Persimmon_Resume Scoring.ipynb

import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd


def preprocess_text(text):
    no_punctuation = [char for char in text if char not in string.punctuation]
    no_punctuation = "".join(no_punctuation)
    words = no_punctuation.lower().split()

    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    processed_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(processed_words)


data = pd.read_excel("../../data/Test Resume and JDs.xlsx")
data = data.dropna()
data["RD"] = data["Description"] + " " + data["Resume Text"]
data["RD"] = data["RD"].apply(preprocess_text)

data["Match Level"] = data["Match Level"].replace("No Match", "Not Match")
data["Match Level"].value_counts()


from imblearn.over_sampling import RandomOverSampler


X = data.drop(columns=["Match Level"])
y = data["Match Level"]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
class_distribution = df_resampled["Match Level"].value_counts()
class_distribution


from imblearn.over_sampling import RandomOverSampler


X = data.drop(columns=["Match Level"])
y = data["Match Level"]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
class_distribution = df_resampled["Match Level"].value_counts()
class_distribution


df_resampled


data.info()


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_resampled["Match Level"])


encoded_classes = label_encoder.classes_
print("Original classes and their encoded values:")
for index, class_label in enumerate(encoded_classes):
    print(f"{index}: {class_label}")


X_train, X_test, y_train, y_test = train_test_split(
    df_resampled["RD"], y, test_size=0.2, random_state=42
)
X_test


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


import xgboost as xgb

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train_vec, y_train)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred_tr = model.predict(X_train_vec)
print(classification_report(y_train, y_pred_tr))


from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, y_pred_tr)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

with open("../model/resume_classifier.pkl", "wb") as file:
    pickle.dump(model, file)

with open("../model/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


def preprocess_text(text):
    no_punctuation = [char for char in text if char not in string.punctuation]
    no_punctuation = "".join(no_punctuation)
    words = no_punctuation.lower().split()

    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    processed_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(processed_words)


with open("../model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("../model/resume_classifier.pkl", "rb") as file:
    model = pickle.load(file)


# def classify_message(message):
# preprocessed_message = preprocess_text(message)
# message_vector = vectorizer.transform([preprocessed_message])
# prediction = model.predict(message_vector)
# return 'Spam' if prediction[0] == 1 else 'Not Spam'


def classify_message(message):
    # Preprocess the input message
    preprocessed_message = preprocess_text(message)

    # Transform the preprocessed message into a vector
    message_vector = vectorizer.transform([preprocessed_message])

    # Predict the class of the message
    prediction = model.predict(message_vector)

    # Get the prediction probabilities
    prediction_probs = model.predict_proba(message_vector)

    # Print the prediction and probabilities for debugging
    print(f"Raw prediction output: {prediction}")
    print(f"Prediction probabilities: {prediction_probs}")

    # Return the predicted class label and probabilities
    predicted_class = prediction[0]
    predicted_class_prob = prediction_probs[0]

    return predicted_class, predicted_class_prob


class_labels = {
    0: "Match",
    1: "Moderate Match",
    2: "No adequate information",
    3: "Not Match",
}


def classify_message(message):

    preprocessed_message = preprocess_text(message)

    message_vector = vectorizer.transform([preprocessed_message])

    prediction = model.predict(message_vector)

    prediction_probs = model.predict_proba(message_vector)
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
