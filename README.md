# PDF Chat with LangChain

This project is a simple Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents. It uses the LangChain framework and OpenAI models to understand and answer questions based on the content of a PDF file.

## Core Functionality

- **PDF Loading**: Extracts text directly from PDF files.
- **Text Cleaning**: Pre-processes the extracted text to remove noisy metadata, improving model performance.
- **Vector Indexing**: Splits the text into chunks and stores them as vector embeddings in a local FAISS store for efficient searching.
- **Question Answering**: Uses a `RetrievalQA` chain to find relevant document sections and generate answers with an OpenAI model.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY="your_openai_api_key_here"
   ```

## How to Use

1. **Configure the PDF Path:**
   In `chat_document.py`, update the `pdf_path` variable to point to your desired PDF file.

2. **Run the Application:**
   ```bash
   python3 chat_document.py
   ```

3. The script will process the PDF and then prompt you to ask questions about its content.