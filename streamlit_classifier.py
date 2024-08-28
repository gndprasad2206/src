import base64
import helpers.classifier_helper as ch
import helpers.data_helper as datah
import helpers.pdf_helper as pdfh
import pandas as pd
import streamlit as st
import os

import requests  # Import requests to send HTTP requests
import streamlit as st


# Function to send JSON data to webhook
def send_json_to_webhook(json_data):
    webhook_url = "https://connect.pabbly.com/workflow/sendwebhookdata/IjU3NjUwNTY0MDYzMzA0MzU1MjY0NTUzNTUxMzEi_pc"
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(webhook_url, json=json_data, headers=headers)
        if response.status_code == 200:
            st.success("JSON data sent to webhook successfully!")
        else:
            st.error(
                f"Failed to send JSON data to webhook. Status code: {response.status_code}"
            )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def streamlit_main():

    # Streamlit UI
    st.title("Resume Matcher")

    # version=st.text_input("Enter the version you want")

    # directory = "../model"
    classifier_directory = ch.get_absolute_path("../../model/classifiers")
    vectorizer_directory = ch.get_absolute_path("../../model/vectorizers")
    # print(directory)

    classifier_model_files = [f for f in os.listdir(classifier_directory) if f.endswith(".pkl")]
    if not classifier_model_files:
        st.error("No model files found in the directory.")
        return
    
    vectorizer_model_files = [f for f in os.listdir(vectorizer_directory) if f.endswith(".pkl")]
    if not vectorizer_model_files:
        st.error("No model files found in the directory.")
        return
    selected_model_classifier = st.selectbox("Select the model version:", classifier_model_files)
    selected_model_vectorizer = st.selectbox(
        "Select the vectorizer version:", vectorizer_model_files
    )
    # print("the selected model is ",selected_model)

    ch.run_once()

    # Input JD
    jd_input = st.text_area("Enter Job Description (JD):")

    # Input Job Title and Company/Client Name
    job_title_input = st.text_input("Enter Job Title:")
    company_name_input = st.text_input("Enter Company/Client Name:")

    # Upload Resumes
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF format)",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_upload",
    )

    if st.button("Match") and jd_input and uploaded_files:
        # Process JD
        # jd_vector, jd_vectorizer = preprocess_and_vectorize([jd_input])

        results = []
        for uploaded_file in uploaded_files:
            # st.write(f"uploaded_file.name: {uploaded_file.name}")
            # st.write(f"uploaded_file.name: {uploaded_file.name}")
            # Read the PDF and extract text
            resume_text = pdfh.get_text_from_stream(uploaded_file)

            if not resume_text:
                print("the resume_text is : ",resume_text)

            message = jd_input + " " + resume_text
            result, probs = ch.classify_message(
                message, selected_model_classifier, selected_model_vectorizer
            )

            # TODO: Convert this to valid json to be stored in the spreadsheet.
            probabilities = ""
            for index, prob in enumerate(probs):
                probabilities += f"{ch.class_labels[index]}: {prob:.4f}; "

            score = probs[ch.get_class_key(result)] * 100

            for prob in probs:
                probabilities += str(prob) + ":"

            # Extract candidate details
            candidate_name = datah.get_candidate_name(resume_text)
            email_address = datah.get_email_from_text(resume_text)

            # Use the inputs provided for Job Title and Company/Client Name
            job_title = job_title_input if job_title_input else "Not provided"
            company_client = (
                company_name_input if company_name_input else "Not provided"
            )

            results.append(
                {
                    "Candidate Name": candidate_name,
                    "Email Address": email_address,
                    "Result": result,
                    "Score": score,
                    "Probabilities": probabilities,
                    "Job Title": job_title,
                    "Company/Client": company_client,
                }
            )

        # Display results
        results_df = pd.DataFrame(results)
        st.write(results_df)

        # Provide export option
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="resume_matches.csv">Export CSV</a>',
            unsafe_allow_html=True,
        )

        # TODO: Enable posting the results to downstream systems using webhook.
        #     # Save the CSV in session state for webhook
        #     st.session_state['csv_data'] = csv

        # # Send CSV data to webhook on export
        # if st.button("Send to Webhook"):
        #     if 'csv_data' in st.session_state:
        #         st.write("inside Send to Webhook")  # This should now appear
        #         send_csv_to_webhook(st.session_state['csv_data'])
        #     else:
        #         st.warning("No CSV data available to send.")
        json_data = results_df.to_dict(orient="records")
        st.session_state["label_json"] = {"response": json_data}

    if st.button("Send to Webhook"):
        if "label_json" in st.session_state:
            send_json_to_webhook(st.session_state["label_json"])
        else:
            st.warning("No JSON data available to send.")


streamlit_main()
