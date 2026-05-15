import streamlit as st


def get_secret(name: str, default: str = None) -> str:
    return st.secrets.get(name, default)


def check_credentials(username: str, password: str) -> bool:
    valid_username = get_secret("username", "test")
    valid_password = get_secret("password", "0123456789p")
    return username == valid_username and password == valid_password


def get_gemini_api_key() -> str:
    return get_secret("gemini_api_key", "")


def get_gemini_model() -> str:
    return get_secret("gemini_model", "gemini-3.5")


def get_gemini_embedding_model() -> str:
    return get_secret("gemini_embedding_model", "gemini-embedding-1.0")
