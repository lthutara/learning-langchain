import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Function to load a single PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# Function to load all PDFs from a directory
def load_pdfs_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Function to clean extracted text from documents
def clean_documents(documents):
    cleaned_docs = []
    noise_pattern = re.compile(r'^.*\.indd.*\d{2}-\d{2}-\d{4}.*$|^Reprint.*$', re.MULTILINE)

    for doc in documents:
        cleaned_content = re.sub(noise_pattern, '', doc.page_content)
        cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))
    return cleaned_docs

# Create or load vector store
def get_or_create_vector_store(doc_path):
    if os.path.isdir(doc_path):
        store_name = os.path.basename(os.path.normpath(doc_path))
    else:
        store_name = os.path.splitext(os.path.basename(doc_path))[0]
    store_path = f"vector_stores/{store_name}.faiss"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(store_path):
        print(f"Loading vector store from {store_path}")
        vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"No existing vector store found. Processing documents...")
        if os.path.isdir(doc_path):
            docs = load_pdfs_from_directory(doc_path)
        else:
            docs = load_pdf(doc_path)
        cleaned_docs = clean_documents(docs)
        split_docs = split_text(cleaned_docs)
        print("Creating new vector store...")
        vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
        print(f"Saving new vector store to {store_path}")
        vectorstore.save_local(store_path)
    return vectorstore

# Ask a question using OpenAI chat model
def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.invoke({"query": question})
    return result
