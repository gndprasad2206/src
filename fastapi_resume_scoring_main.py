from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Optional
import pandas as pd
import base64
import helpers.classifier_helper as ch
import helpers.data_helper as datah
import helpers.pdf_helper as pdfh
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow requests from your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.post("/process-resumes/")
async def process_resumes(
    job_description: str = Form(...),
    job_title: str = Form(...),
    company_name: Optional[str] = Form(None),
    classifier_version: str = Form(...),
    vectorizer_version: str = Form(...),
    files: List[UploadFile] = File(...)
):

    selected_model_classifier = classifier_version
    selected_model_vectorizer = vectorizer_version

    ch.run_once()

    results = []
    for file in files:
        # Read the PDF and extract text
        resume_text = pdfh.get_text_from_stream(file.file)

        if not resume_text:
            continue

        message = job_description + " " + resume_text
        result, probs = ch.classify_message(
            message, selected_model_classifier, selected_model_vectorizer
        )

        # Extract candidate details
        candidate_name = datah.get_candidate_name(resume_text)
        email_address = datah.get_email_from_text(resume_text)

        # Construct result
        probabilities = "; ".join(
            [f"{ch.class_labels[i]}: {prob:.4f}" for i, prob in enumerate(probs)]
        )
        score = probs[ch.get_class_key(result)] * 100

        results.append({
            "Candidate Name": candidate_name,
            "Email Address": email_address,
            "Result": result,
            "Score": score,
            "Probabilities": probabilities,
            "Job Title": job_title,
            "Company/Client": company_name or "Not provided"
        })

    # Convert results to DataFrame and JSON
    results_df = pd.DataFrame(results)
    csv_data = results_df.to_csv(index=False)
    json_data = results_df.to_dict(orient="records")

    # Encode CSV to base64 for download
    b64_csv = base64.b64encode(csv_data.encode()).decode()

    return {
        "csv": b64_csv,
        "json": json_data
    }
