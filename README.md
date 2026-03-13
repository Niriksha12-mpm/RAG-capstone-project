# RAG Document Question Answering System

A Retrieval Augmented Generation (RAG) based application that allows users to ask questions from PDF documents.

## Overview

This project demonstrates the implementation of a RAG pipeline where information is retrieved from documents and used by a Large Language Model to generate accurate answers.

The system supports both preloaded PDFs and user-uploaded documents.

## Features

- Upload PDF documents
- Extract text from documents
- Semantic search using embeddings
- AI-generated answers based on document content
- Support for multiple questions

## Technologies Used

- Python
- Groq API
- Retrieval Augmented Generation (RAG)
- PDF processing libraries
- Vector embeddings

## System Workflow

1. Upload a PDF document.
2. Extract text from the document.
3. Convert text into embeddings.
4. Retrieve relevant document sections using semantic search.
5. Use LLM to generate answers based on retrieved context.

## Installation

Clone the repository
git clone https://github.com/Niriksha12-mpm/rag-document-qa.git�

Navigate to the folder
cd rag-document-qa

Install dependencies
pip install -r requirements.txt

Run the application
python app.py

## Future Improvements

- Support multiple document collections
- Web-based interface
- Chat-style interaction with documents
- Document summarization

## Author

Niriksha M P M
