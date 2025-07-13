# Project TODO List

Here are the planned features and improvements for this RAG application.

- [ ] **Optimize Embedding Pipeline:**
  - [ ] Check if a FAISS index file exists on startup.
  - [ ] If it exists, load the vector store from disk.
  - [ ] If it does not exist, create the embeddings and save the new vector store to disk.

- [ ] **Add Multi-Document Support:**
  - [ ] Modify the script to accept a folder path instead of a single file path.
  - [ ] Iterate through all PDF files in the given folder.
  - [ ] Load and process all documents into a single, combined vector store.

- [ ] **Implement a Graphical User Interface (GUI):**
  - [ ] Choose a framework (e.g., Streamlit, Gradio).
  - [ ] Replace the command-line `input()` loop with a web-based interface for asking questions.
  - [ ] Display the answers in the web interface.
