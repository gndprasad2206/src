from helpers import data_helper, ml_helper
from helpers import pdf_helper


def main():
    dh = data_helper.DataHelper()
    pdfh = pdf_helper.PdfHelper()
    # label_df = dh.load_labels('/Users/vnagireddy/repos/git/github.com/symphonize/persimmon-ai/prototypes/ml/data/Persimmon Labelling.xlsx', 'JD1')
    label_df = dh.load_labels('/Users/vnagireddy/repos/git/github.com/symphonize/persimmon-ai/prototypes/ml/data/Test Resume and JDs.xlsx', 'Sheet1')

    # df = dh.get_data_frame()
    # df = dh.transform_data_frame(label_df)
    df = dh.transform_data_frame_v0_20240805141000_pst(label_df)
    mlh = ml_helper.MLHelper()
    X_combined, y = mlh.extract_features(df)
    mlh.train_model(X_combined, y)

    # Example prediction
    new_resume_text = "Skills: Python, Machine Learning, Data Analysis, SQL. Experience: Data Scientist at XYZ Corp for 3 years, Software Engineer at ABC Inc for 2 years."
    new_job_description_text = "Requirements: Proficiency in Python, experience with machine learning and data analysis, knowledge of SQL, 3+ years of relevant experience."
    print(new_resume_text)
    print(new_job_description_text)
    predicted_score = mlh.predict_score(new_resume_text, new_job_description_text)
    print(f'Predicted Match Score: {predicted_score}')

if __name__ == '__main__':
    main()
