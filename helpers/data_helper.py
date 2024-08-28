import re

# Function to extract candidate's name from resume text
def get_candidate_name(resume_text):
    lines = resume_text.split("\n")
    name_line = lines[0]  # Assuming the name is the first line; adjust as needed
    return name_line.strip()


# Function to extract email address from resume text
def get_email_from_text(resume_text):
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_regex, resume_text)
    if emails:
        return emails[0]  # Return the first found email address
    return "Not provided"
