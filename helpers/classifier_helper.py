import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

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


def load_model(selected_path):
    # MODEL_PATH = "../../model/resume_classifier.pkl"
    # MODEL_PATH = os.path.join("..", "..", "..", "model", "resume_classifier.pkl")
    # print("the slected path at the load model is : ",selected_path)
    d1_path = "../../model/"+selected_path+""
    # print("the constructed path is : ",d1_path)
    # d_path = "../../model/resume_classifier"+v+".pkl"
    # print(d_path)
    # path="../../model/resume_classifier-v0.001-03.pkl"
    # if d1_path==path:
    #     print("proceed")
    # else:
    #     print("not proceed")
    MODEL_PATH = get_absolute_path(d1_path)

    # print("the model path is : ",MODEL_PATH)

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def load_vectorizer(selected_path):
    # VECTORIZER_PATH = "../../model/vectorizer.pkl"
    # VECTORIZER_PATH = os.path.join("..", "..", "..", "model", "vectorizer.pkl")
    # VECTORIZER_PATH = get_absolute_path("../../../model/vectorizer.pkl")
    d1_path = "../../model/"+selected_path+""

    # VECTORIZER_PATH = get_absolute_path("../../model/vectorizer.pkl")
    VECTORIZER_PATH = get_absolute_path(d1_path)


    with open(VECTORIZER_PATH, "rb") as file:
        return pickle.load(file)


class_labels = {
    0: "Match",
    1: "Moderate Match",
    2: "No adequate information",
    3: "Not Match",
}

get_class_key = lambda value: next((k for k, v in class_labels.items() if v == value), None)

def classify_message(message,model_version,vectorizer_version):
    preprocessed_message = preprocess_text(message)
    loaded_model = load_model(model_version)
    loaded_vectorizer = load_vectorizer(vectorizer_version)
    message_vector = loaded_vectorizer.transform([preprocessed_message])
    prediction = loaded_model.predict(message_vector)
    prediction_probs = loaded_model.predict_proba(message_vector)
    predicted_class_label = class_labels[prediction[0]]
    # print(f"Raw prediction output: {prediction[0]} ({predicted_class_label})")
    # print(f"Prediction probabilities: {prediction_probs[0]}")
    return predicted_class_label, prediction_probs[0]


