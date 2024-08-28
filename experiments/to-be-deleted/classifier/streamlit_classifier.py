import base64
import classifier_helper as ch
import data_helper as datah
import pandas as pd
import pdf_helper as pdfh
import streamlit as st


def streamlit_main():
    # Streamlit UI
    st.title("Resume Matcher")

    ch.run_once()

    # Input JD
    jd_input = st.text_area("Enter Job Description (JD):")

    # Input Job Title and Company/Client Name
    job_title_input = st.text_input("Enter Job Title:")
    company_name_input = st.text_input("Enter Company/Client Name:")


    # Upload Resumes
    uploaded_files = st.file_uploader("Upload Resumes (PDF format)", type=["pdf"], accept_multiple_files=True, key="resume_upload")

    if st.button("Match") and jd_input and uploaded_files:
        # Process JD
        # jd_vector, jd_vectorizer = preprocess_and_vectorize([jd_input])

        results = []
        for uploaded_file in uploaded_files:
            # st.write(f"uploaded_file.name: {uploaded_file.name}")
            # st.write(f"uploaded_file.name: {uploaded_file.name}")
            # Read the PDF and extract text
            resume_text = pdfh.get_text_from_stream(uploaded_file)

            message = jd_input + " " + resume_text
            result, probs = ch.classify_message(message)

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
            company_client = company_name_input if company_name_input else "Not provided"

            results.append({
                "Candidate Name": candidate_name,
                "Email Address": email_address,
                "Result": result,
                "Score": score,
                "Probabilities": probabilities,
                "Job Title": job_title,
                "Company/Client": company_client
            })

        # Display results
        results_df = pd.DataFrame(results)
        st.write(results_df)

        # Provide export option
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="resume_matches.csv">Export CSV</a>', unsafe_allow_html=True)

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

streamlit_main()

