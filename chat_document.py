from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import re
from langchain.schema import Document

load_dotenv()

# Now the API key will be read from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Function to load and split PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Function to clean extracted text from documents
def clean_documents(documents):
    cleaned_docs = []
    # Regex to find and remove lines that look like file paths or printing metadata
    # This pattern looks for '.indd' and a date, or 'Reprint'
    noise_pattern = re.compile(r'^.*\.indd.*\d{2}-\d{2}-\d{4}.*$|^Reprint.*$', re.MULTILINE)

    for doc in documents:
        cleaned_content = re.sub(noise_pattern, '', doc.page_content)
        # We create a new Document object to keep the same structure
        cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))
    return cleaned_docs

# Ask a question using OpenAI chat model
def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.invoke({"query": question})
    return result

# Main function
def main(pdf_path):
    # Create a path for the vector store based on the PDF file name
    # The store will be created in the same directory as the script.
    store_name = os.path.splitext(os.path.basename(pdf_path))[0]
    store_path = f"{store_name}.faiss"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(store_path):
        print(f"Loading vector store from {store_path}")
        vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"No existing vector store found for {pdf_path}. Processing PDF...")
        docs = load_pdf(pdf_path)
        cleaned_docs = clean_documents(docs)
        split_docs = split_text(cleaned_docs)
        print("Creating new vector store...")
        vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
        print(f"Saving new vector store to {store_path}")
        vectorstore.save_local(store_path)

    while True:
        question = input("\nAsk a question about the PDF (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = ask_question(vectorstore, question)
        print(f"\nAnswer:\n{answer}\n")

# Example usage
if __name__ == "__main__":
    pdf_path = "/home/lthutara/learning-langchain/ganitha-prakash/fegp110.pdf"  # Replace with your PDF path
    main(pdf_path)
