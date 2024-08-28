import math
import os
import pandas as pd
import PyPDF2

from helpers.pdf_helper import PdfHelper

class DataHelper:
    def __init__(self):
        pass


    def get_data_frame(self):
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

        print(data)

        return pd.DataFrame(data)
  

    def load_resumes(self, resume_folder):
        resumes = []
        for file in os.listdir(resume_folder):
            if file.endswith(".pdf"):
                with open(os.path.join(resume_folder, file), 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text()

            resumes.append({'resume_id': file, 'text': text})
        return pd.DataFrame(resumes)


    def load_job_descriptions(self, job_description_folder):
        pass


    def load_labels(self, workbook, sheet):
        return pd.read_excel(workbook, sheet_name=sheet)
    
    def get_resume_text(self, resume):
        pass

    def get_match_score(self, label):
        if label == 'Match':
            return 1.0
        elif label == 'Moderate Match':
            return 0.5
        else:
            return 0.0

    def transform_data_frame(self, input_data_frame):
        for index, row in input_data_frame.iterrows():
            jd_tmp = row['JD1']
            if isinstance(jd_tmp, str) and len(jd_tmp) > 0:
                job_description = jd_tmp
        output_data = []
        for index, row in input_data_frame.iterrows():
            resume_file_path = f"/Users/vnagireddy/root/var/data/persimmon/resumes/{row['Resume']}.pdf"
            resume_text = PdfHelper().get_text(resume_file_path)
            if resume_text != None:
                output_data.append({'resume_file_path': resume_file_path, 'resume_text': PdfHelper().get_text(resume_file_path), 'job_description_text': job_description, 'match_score': self.get_match_score(row['LABEL(avg)'])})
            # output_data_frame['resume_text'] = PdfHelper().get_text(resume_file_path)
        output_data_frame = pd.DataFrame(output_data)

        print(output_data_frame)
        return output_data_frame

    def transform_data_frame_v0_20240805141000_pst(self, input_data_frame):
        output_data = []
        for _, row in input_data_frame.iterrows():
            job_title = row['Title']
            job_description = row['Description']
            resume_text = row['Resume Text']
            if resume_text != None and job_description != None:
                output_data.append(
                    {
                        'job_title': job_title,
                        'resume_text': resume_text,
                        'job_description_text': job_description,
                        'match_score': self.get_match_score(row['Match Level'])
                    }
                )
        output_data_frame = pd.DataFrame(output_data)
        print(output_data_frame)

        return output_data_frame


# TODO: Add unit tests for testing rather than relying on main methods.
"""
def main():
    data_helper = DataHelper()
    pdf_helper = PdfHelper()
    label_df = data_helper.load_labels('/Users/vnagireddy/repos/git/github.com/symphonize/persimmon-ai/prototypes/ml/data/Persimmon Labelling.xlsx', 'JD1')
    print(label_df['Resume'])
    for index, row in label_df.iterrows():
        resume_file_path = f"/Users/vnagireddy/root/var/data/persimmon/resumes/{row['Resume']}.pdf"
        # print(resume_file_path)
        # print(pdf_helper.get_text(resume_file_path))
        # print('/Users/vnagireddy/root/var/data/persimmon/resumes/' + row['Resume'] + '.pdf')
    # print(label_df['LABEL(avg)'])
    # resume_df = data_helper.load_resumes("/Users/vnagireddy/root/var/data/persimmon/resumes")
    # print(resume_df)
    data_helper.transform_data_frame(label_df)
    return
    job_description_df = data_helper.load_job_descriptions("/Users/vnagireddy/root/var/data/persimmon/job-descriptions")
    print('here')
    print(resume_df)
    print(job_description_df)


if __name__ == "__main__":
    main()
"""

