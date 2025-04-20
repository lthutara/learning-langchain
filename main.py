from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import textwrap


load_dotenv()

# Now the API key will be read from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API key
#os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your actual API key

# Function to load and split PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def print_answer(answer):
    # If answer is a dictionary, extract the 'text' field
    if isinstance(answer, dict):
        answer = answer.get('text', 'No answer text found')

    # Wrapping the answer text to 80 characters per line for readability
    wrapped_answer = textwrap.fill(answer, width=80)
    print(f"\nAnswer:\n{wrapped_answer}\n")


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Create vector store with OpenAI embeddings
def create_embeddings(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

# Ask a question using OpenAI chat model
def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.invoke({"query": question})
    return result

# Main function
def main(pdf_path, question):
    docs = load_pdf(pdf_path)
    split_docs = split_text(docs)
    vectorstore = create_embeddings(split_docs)
    while True:
        question = input("\nAsk a question about the PDF (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = ask_question(vectorstore, question)
        #print_answer(answer)
        print(f"\nAnswer:\n{answer}\n")

# Example usage
if __name__ == "__main__":
    pdf_path = "/home/lthutara/ai-learnings/langchain-playground/ind_geo.pdf"  # Replace with your PDF path
    question = "What is this document about?"  # Replace with your question
    main(pdf_path, question)

