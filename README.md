# Chat with Multi PDF using LangChain and Google's Gemini Model

This project enables users to chat with PDF files using LangChain, Google's Gemini model, and Streamlit. The app extracts text from PDF files, processes them into chunks, stores the embeddings in a FAISS vector store, and allows users to ask questions. The Gemini model retrieves and answers questions based on the content from the PDF.

## Project Overview

- **LangChain**: Used for text processing, chunking, and managing the interaction with language models.
- **Google Gemini Model**: Powers the conversational question-answering system.
- **FAISS**: Stores text embeddings for efficient similarity search.
- **Streamlit**: Provides the user interface for uploading PDF files and asking questions.

## Features

- **PDF File Upload**: Users can upload multiple PDF files.
- **Text Chunking**: Automatically splits large PDF text into manageable chunks.
- **Embeddings**: Stores vector embeddings using Google's Gemini model.
- **Natural Language Questions**: Allows users to ask questions in plain English, with responses based on the PDF content.
- **Conversational Interface**: Uses a question-answering chain with a detailed context-based response system.

## Workflow

### 1. **Upload PDFs**:
   - Users upload PDF files through the sidebar using Streamlit's file uploader.
   
### 2. **Text Extraction**:
   - The app uses PyPDF2 to extract text from the uploaded PDF files.

### 3. **Text Chunking**:
   - LangChain's `RecursiveCharacterTextSplitter` splits the extracted text into smaller, manageable chunks for processing. This is essential for handling large documents and ensures context is preserved.

### 4. **Embedding Generation**:
   - The chunks are converted into vector embeddings using Google's Generative AI Embeddings model (`embedding-001`). These embeddings represent the semantic meaning of the text.

### 5. **Vector Store Creation**:
   - FAISS stores the embeddings to perform efficient similarity searches. The vector store is saved locally, allowing fast lookups.

### 6. **Question Input**:
   - Users input natural language questions into the app, which are processed by the Google Gemini model.

### 7. **Answer Retrieval**:
   - Based on the input question, FAISS performs a similarity search on the stored embeddings to find the most relevant text chunks. These chunks are then passed to the Google Gemini model to generate a detailed, context-based response.

### 8. **Answer Display**:
   - The app displays the response to the user in real-time, directly answering the question using content from the uploaded PDFs.

#Snapshot - User Interface
![interface](https://github.com/user-attachments/assets/7859b33b-f4f4-46ba-a631-26eb75a0a5ee)

## Conclusion

This project provides a robust solution for interacting with documents using advanced natural language processing techniques. By leveraging LangChain and Google’s Gemini model, users can engage with their PDFs conversationally, extracting meaningful insights and answers. The use of embeddings and FAISS for similarity search makes the application efficient, even with large amounts of data. The project demonstrates how language models can simplify information retrieval from documents, making it a valuable tool for academic research, business reports, or personal document management
