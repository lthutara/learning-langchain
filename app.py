import streamlit as st
import os
from dotenv import load_dotenv
from rag_core import get_or_create_vector_store, ask_question

load_dotenv() # Load environment variables at the very beginning

st.set_page_config(layout="wide")
st.title("Multi-Document RAG System")

st.write("Welcome to the RAG system. Please select your options and ask a question.")

# Define the base path for your documents
BASE_DOC_PATH = "/home/lthutara/learning-langchain/data"

# Placeholders for future UI elements
subject_selection = st.selectbox("Select Subject", ["Mathematics", "Science", "Social Studies"])
class_selection = st.selectbox("Select Class", [f"Class_{i}" for i in range(6, 11)])

question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if question:
        # Construct the document path based on selections
        doc_path = os.path.join(BASE_DOC_PATH, class_selection, subject_selection)

        if not os.path.exists(doc_path):
            st.error(f"Document path not found: {doc_path}. Please ensure the data is organized correctly.")
        else:
            with st.spinner(f"Loading or creating vector store for {subject_selection} - {class_selection}..."):
                vectorstore = get_or_create_vector_store(doc_path)

            if vectorstore:
                with st.spinner("Getting answer..."):
                    answer = ask_question(vectorstore, question)
                st.success("Answer:")
                st.write(answer["result"])
            else:
                st.error("Failed to load or create vector store.")
    else:
        st.warning("Please enter a question.")