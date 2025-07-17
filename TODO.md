# Project TODO List

Here are the planned features and improvements for this RAG application.

- [x] **Optimize Embedding Pipeline:**
  - [x] Check if a FAISS index file exists on startup.
  - [x] If it exists, load the vector store from disk.
  - [x] If it does not exist, create the embeddings and save the new vector store to disk.

- [x] **Add Multi-Document Support:**
  - [x] Modify the script to accept a folder path instead of a single file path.
  - [x] Iterate through all PDF files in the given folder.
  - [x] Load and process all documents into a single, combined vector store.

- [x] **Implement a Graphical User Interface (GUI):**
  - [x] Choose a framework (e.g., Streamlit, Gradio).
  - [x] Replace the command-line `input()` loop with a web-based interface for asking questions.
  - [x] Display the answers in the web interface.

- [ ] **GUI Improvements:**
  - [ ] Make the document folder path dynamic/configurable in the Streamlit app.
  - [ ] Enhance the question-asking flow (e.g., clear input field after submission, allow continuous questioning without re-clicking).
  - [ ] Improve the overall aesthetic and user experience of the Streamlit GUI.