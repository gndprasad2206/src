# TODO: move to common/helpers/pdf_helper.py

import PyPDF2
import os


def get_text_from_filepath(filepath):
    if filepath.endswith(".pdf") and os.path.exists(filepath):
        with open(filepath, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text


def get_text_from_stream(stream):
    text = ""
    pdf_reader = PyPDF2.PdfReader(stream)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

