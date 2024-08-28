from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class MLHelper:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.num_features = None


    def extract_features(self, df):
        # Handle NaN values by filling them with an empty string
        df['resume_text'] = df['resume_text'].fillna('')
        df['job_description_text'] = df['job_description_text'].fillna('')

        combined_text = df['resume_text'] + ' ' + df['job_description_text']
        X = self.vectorizer.fit_transform(combined_text).toarray()

        self.num_features = X.shape[1] // 2
        X_resume = X[:, :self.num_features]
        X_job_desc = X[:, self.num_features:]

        X_combined = X_resume * X_job_desc
        y = df['match_score']
        return X_combined, y


    def train_model(self, X_combined, y):
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')


    def predict_score(self, resume_text, job_description_text):
        combined_text = resume_text + ' ' + job_description_text
        X_new = self.vectorizer.transform([combined_text]).toarray()
        X_new_resume = X_new[:, :self.num_features]
        X_new_job_desc = X_new[:, self.num_features:]
        X_new_combined = X_new_resume * X_new_job_desc
        return self.model.predict(X_new_combined)[0]
    
    
    def extract_skills(self, df):
        pass


    def extract_experience(self, df):
        pass
    
