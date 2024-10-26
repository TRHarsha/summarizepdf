import streamlit as st
from pdfminer.high_level import extract_text
from transformers import pipeline

# Initialize the Hugging Face summarization pipeline with a different model
summarizer = pipeline("summarization", model="t5-small")  # You can change this to another model

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using PDFMiner."""
    return extract_text(pdf_file)

def summarize_text(text):
    """Summarize the extracted text using Hugging Face pipeline."""
    max_chunk_size = 512
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    summarized_text = ""
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=100, min_length=25, do_sample=False)
        summarized_text += summary[0]["summary_text"] + " "
    return summarized_text

# Streamlit app layout
st.title("PDF Summarizer")
st.write("Upload a PDF file, and this tool will summarize its contents.")

uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_pdf is not None:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_pdf)
        st.write("### Extracted Text")
        st.write(extracted_text[:1000] + "...")  # Display the first 1000 characters for reference

    if st.button("Summarize Text"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(extracted_text)
            st.write("### Summary")
            st.write(summary)
