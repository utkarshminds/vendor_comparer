# vendor_comparer
Vendor comparer

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run login.py
```

## Login Credentials

For local development, credentials are stored in `.streamlit/secrets.toml`:
- **Username:** test
- **Password:** 0123456789p

## Deployment on Streamlit Cloud

1. Push your code to GitHub (excluding `.streamlit/secrets.toml`)
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Create a new app and select your repository
4. In the app settings, navigate to **Secrets** and add:
```toml
username = "your_username"
password = "your_password"
```
5. Deploy the app

The app will automatically read credentials from Streamlit secrets during deployment.
