# vendor_comparer
Quotation comparison and file chat system

## Features

- **Secure login** with username and password stored in Streamlit secrets
- **Quotation upload guardrails**: only quotation documents are accepted
- **File analysis**: automatically compare uploaded quotation documents and summarize pros and cons
- **Chat interface**: ask questions and receive answers derived only from uploaded files
- **File management**: download or delete uploaded quotation files
- **Gemini integration**: secure API key stored in secrets

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

3. Run the Streamlit app:
```bash
streamlit run login.py
```

## Usage

1. Login with your credentials
2. Upload quotation files in `.txt` or `.md` format
3. The app validates that the file contains a quotation document
4. The app analyzes uploaded quotations and compares pros and cons
5. Ask questions in the chat box and receive answers sourced only from uploaded files
6. Delete uploaded files from the dashboard when needed

## Streamlit Cloud Deployment

1. Push the code to GitHub. Do not commit `.streamlit/secrets.toml` or the `uploads/` folder.
2. Create a new app on [Streamlit Cloud](https://share.streamlit.io/).
3. In app settings, add the following secrets:
```toml
username = "your_username"
password = "your_password"
gemini_api_key = "YOUR_GEMINI_API_KEY"
```
4. Deploy the app.

The app reads credentials and Gemini secrets from Streamlit secrets at runtime.

## Security

- Secrets are stored in `.streamlit/secrets.toml`
- `.streamlit/secrets.toml` and `uploads/` are ignored via `.gitignore`
- The app rejects files that are not recognized as quotations
- Off-topic queries are blocked by the chat prompt guardrail

