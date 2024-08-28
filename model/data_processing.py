def preprocess_text(text):
    no_punctuation = [char for char in text if char not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)
    words = no_punctuation.lower().split()

    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    processed_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)
