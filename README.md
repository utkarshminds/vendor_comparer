# vendor_comparer
Quotation comparison and file chat system

## Overview

`vendor_comparer` is a Streamlit-based application for securely uploading quotation documents, validating that they are actual quotations, comparing them with pros and cons summaries, and allowing users to chat with the uploaded files. The application is designed using a modular architecture, with clear separation between authentication, storage, Gemini API integration, and quotation reasoning.

## Documentation

### `login.py`
- Main Streamlit app entry point
- Handles authentication, file upload flow, file listing, deletion, analysis summary, and chat interface
- Uses session state to preserve uploaded document context and conversation state

### `auth.py`
- Centralizes secure secrets retrieval
- Provides login credential verification
- Retrieves Gemini API configuration from `st.secrets`

### `storage.py`
- Manages file storage and file guardrails
- Creates `uploads/` directory
- Saves uploaded `.txt`, `.md`, and `.pdf` files
- Loads and deletes uploaded files
- Ensures only supported document types are accepted

### `gemini_client.py`
- Encapsulates Gemini API requests for text generation and embeddings
- Protects the API key by reading it from `st.secrets`
- Uses direct HTTPS POST requests to Gemini endpoints

### `quote_system.py`
- Validates uploaded documents as quotations
- Builds embeddings for document retrieval
- Generates comparative pros and cons summaries
- Answers user questions using only uploaded quotation content
- Contains guardrails for off-topic queries and quotation-only responses

## Features

- Secure user authentication via Streamlit secrets
- Quotation upload guardrails to reject unrelated files
- Optional permanent storage in vector database for persistent access
- Immediate document validation before saving
- Persistent vector database storage for quotation chunks
- Retrieval Augmented Generation (RAG) for chat answers from uploaded quotations
- Automatic quotation comparison and pros/cons analysis
- Auto-generate comparison table with parameters as rows and vendors as columns
- Chat interface scoped to uploaded file content only
- File management with download and delete functionality
- Modular design for maintainability and security

## Algorithm and Workflow

1. **Login**
   - User provides credentials stored in `.streamlit/secrets.toml`
   - The app validates credentials and stores login state in Streamlit session state

2. **Upload and validation**
   - The user can upload `.txt`, `.md`, or `.pdf` files
   - Each file is decoded or extracted and validated using the Gemini model as a quotation document
   - Non-quotation files are rejected with a warning message
   - User can optionally save files to permanent vector database storage

3. **Storage**
   - Valid quotation files are written to `uploads/` for session access
   - If permanent storage is selected, files are chunked and stored in a local vector database for persistent retrieval
   - Filenames are tracked in session state with their parsed text content
   - Uploaded files can be deleted at any time from both session and permanent storage

4. **Vector DB and retrieval**
   - Uploaded quotation files are split into overlapping text chunks
   - Each chunk is embedded with Gemini and stored in a local vector database
   - The vector DB is queried for the most relevant chunks when the user asks a question

5. **Comparison summary**
   - The app sends uploaded quotation content to Gemini with a prompt that asks for pros, cons, risks, pricing notes, and notable terms
   - The result is displayed as a consolidated analysis summary

6. **Auto-generate table**
   - User can click "Auto-generate Comparison Table" to create a structured table
   - Table has parameters (Pricing, Terms, etc.) as rows and vendor names as columns
   - Each cell contains specific information extracted from the documents

7. **Chat with files**
   - User questions are answered using only the uploaded documents
   - Gemini receives the top relevant document excerpts and a strict prompt enforcing source-only responses
   - Off-topic questions are rejected with a fixed failure message

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure local secrets in `.streamlit/secrets.toml`:
```toml
username = "test"
password = "0123456789p"
gemini_api_key = "YOUR_GEMINI_API_KEY"
# Optional:
gemini_model = "gemini-1.0"
gemini_embedding_model = "gemini-embedding-1.0"
```

3. Run the app:
```bash
streamlit run login.py
```

## Deployment Notes

- Do not commit `.streamlit/secrets.toml`, the `uploads/` folder, or the `vector_db/` folder to GitHub.
- Add `gemini_api_key` in Streamlit Cloud secrets before deployment.
- The app is designed so that answers are only based on uploaded quotation documents.

## Security and Best Practices

- Secrets are stored in `.streamlit/secrets.toml` and not committed
- `uploads/` and `vector_db/` are ignored in `.gitignore`
- Quotation validation prevents unrelated or malicious file content from entering the system
- Chat responses are restricted to the uploaded files via prompt guardrails

## Supported File Types

- `.txt`
- `.md`
- `.pdf`

## Notes

- This project uses Gemini for both text generation and embeddings.
- The application is intentionally limited to quotation documents and will decline unrelated requests.

