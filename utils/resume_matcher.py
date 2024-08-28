# Colab File: Resume Matching with Job Descriptions using Machine Learning

# Step 1: Install Required Libraries
!pip install nltk spacy sklearn
!python -m spacy download en_core_web_sm

# Step 2: Prepare the Dataset
import pandas as pd

# Sample data
data = {
    'resume_text': [
        "Skills: Python, Machine Learning, Data Analysis, SQL. Experience: Data Scientist at XYZ Corp for 3 years, Software Engineer at ABC Inc for 2 years.",
        "Skills: Java, Spring, Hibernate. Experience: Software Engineer at DEF Corp for 5 years.",
        # Add more resumes
    ],
    'job_description_text': [
        "Requirements: Proficiency in Python, experience with machine learning and data analysis, knowledge of SQL, 3+ years of relevant experience.",
        "Requirements: Experience in Java, Spring framework, Hibernate ORM, 4+ years of relevant experience.",
        # Add more job descriptions
    ],
    'match_score': [1, 0.8]  # Sample match scores
}

df = pd.DataFrame(data)

# Step 3: Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine resume and job description text for vectorization
combined_text = df['resume_text'] + ' ' + df['job_description_text']

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(combined_text).toarray()

# Split the features into resume and job description parts
num_features = X.shape[1] // 2
X_resume = X[:, :num_features]
X_job_desc = X[:, num_features:]

# Combine resume and job description features
X_combined = X_resume * X_job_desc  # Element-wise multiplication
y = df['match_score']

# Step 4: Train the Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 5: Use the Model for Predictions
def predict_match_score(resume_text, job_description_text):
    combined_text = resume_text + ' ' + job_description_text
    X_new = vectorizer.transform([combined_text]).toarray()
    X_new_resume = X_new[:, :num_features]
    X_new_job_desc = X_new[:, num_features:]
    X_new_combined = X_new_resume * X_new_job_desc
    return model.predict(X_new_combined)[0]

# Example prediction
new_resume_text = "Skills: Python, Machine Learning, Data Analysis, SQL. Experience: Data Scientist at XYZ Corp for 3 years, Software Engineer at ABC Inc for 2 years."
new_job_description_text = "Requirements: Proficiency in Python, experience with machine learning and data analysis, knowledge of SQL, 3+ years of relevant experience."
predicted_score = predict_match_score(new_resume_text, new_job_description_text)
print(f'Predicted Match Score: {predicted_score}')
