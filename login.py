import os
import streamlit as st

from auth import (
    check_credentials,
    get_gemini_api_key,
    get_gemini_model,
    get_gemini_embedding_model,
)
from gemini_client import GeminiClient
from quote_system import (
    answer_from_vector_db,
    compare_quotes,
    generate_comparison_table,
    validate_quotation_document,
)
from storage import (
    allowed_file_type,
    create_uploads_directory,
    delete_uploaded_file,
    save_uploaded_file,
)
from vector_db import VectorDB


def init_session_state() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "documents" not in st.session_state:
        st.session_state.documents = {}
    if "analysis_summary" not in st.session_state:
        st.session_state.analysis_summary = ""
    if "chat_response" not in st.session_state:
        st.session_state.chat_response = ""
    if "upload_warnings" not in st.session_state:
        st.session_state.upload_warnings = []
    if "table_response" not in st.session_state:
        st.session_state.table_response = ""


def get_gemini_client() -> GeminiClient | None:
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    return GeminiClient(
        api_key=api_key,
        model=get_gemini_model(),
        embed_model=get_gemini_embedding_model(),
    )


def get_vector_db(client: GeminiClient) -> VectorDB:
    return VectorDB(client)


def upload_and_validate_files(uploaded_files, client: GeminiClient, vector_db: VectorDB, save_permanent: bool) -> None:
    if not uploaded_files:
        return

    create_uploads_directory()
    new_documents = {}
    st.session_state.upload_warnings = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        if not allowed_file_type(filename):
            st.session_state.upload_warnings.append(
                f"Skipping {filename}: unsupported file type. Only .txt and .md files are allowed."
            )
            continue

        if filename in st.session_state.documents:
            st.session_state.upload_warnings.append(
                f"Skipping {filename}: file already uploaded."
            )
            continue

        file_text = uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()
        if not file_text:
            st.session_state.upload_warnings.append(
                f"Skipping {filename}: file contains no readable text."
            )
            continue

        try:
            valid = validate_quotation_document(file_text, client)
        except Exception as exc:
            st.session_state.upload_warnings.append(
                f"Unable to validate {filename}: {exc}"
            )
            continue

        if not valid:
            st.session_state.upload_warnings.append(
                f"Skipping {filename}: document is not recognized as a quotation."
            )
            continue

        saved_name = save_uploaded_file(uploaded_file)
        new_documents[saved_name] = file_text
        if save_permanent:
            vector_db.add_document(saved_name, file_text)

    if new_documents:
        st.session_state.documents.update(new_documents)
        st.session_state.analysis_summary = compare_quotes(
            st.session_state.documents, client
        )
        st.success(f"Uploaded {len(new_documents)} quotation file(s) successfully.")


def delete_file(filename: str, client: GeminiClient, vector_db: VectorDB) -> None:
    if delete_uploaded_file(filename):
        st.session_state.documents.pop(filename, None)
        vector_db.delete_document(filename)
        if st.session_state.documents:
            st.session_state.analysis_summary = compare_quotes(
                st.session_state.documents, client
            )
        else:
            st.session_state.analysis_summary = ""
            st.session_state.table_response = ""
        st.success(f"Deleted {filename} successfully.")
        st.experimental_rerun()


def render_uploaded_files(client: GeminiClient, vector_db: VectorDB) -> None:
    if not st.session_state.documents:
        return

    st.divider()
    st.subheader("✅ Uploaded Quotations")

    for idx, filename in enumerate(list(st.session_state.documents.keys()), start=1):
        file_path = f"uploads/{filename}"
        size_kb = 0
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024

        cols = st.columns([4, 1, 1])
        cols[0].write(f"{idx}. {filename}")
        cols[1].write(f"{size_kb:.2f} KB")

        if cols[2].button("Delete", key=f"delete_{idx}"):
            delete_file(filename, client, vector_db)

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                cols[2].download_button(
                    label="Download",
                    data=f.read(),
                    file_name=filename,
                    key=f"download_{idx}_button",
                )


def render_chat_panel(client: GeminiClient, vector_db: VectorDB) -> None:
    if not st.session_state.documents:
        return

    st.divider()
    st.subheader("💬 Chat with uploaded quotations")

    with st.form("chat_form"):
        query = st.text_input(
            "Ask a question about the uploaded quotations",
            key="chat_query",
        )
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        if not query:
            st.error("Please enter a question before submitting.")
        else:
            try:
                response = answer_from_vector_db(
                    query,
                    vector_db,
                    client,
                )
                st.session_state.chat_response = response
            except Exception as exc:
                st.error(f"Unable to answer the query: {exc}")

    if st.session_state.chat_response:
        st.markdown("**Response:**")
        st.write(st.session_state.chat_response)


def main() -> None:
    st.set_page_config(page_title="Quotation Comparison", layout="wide")
    init_session_state()

    if st.session_state.logged_in:
        client = get_gemini_client()
        if not client:
            st.error(
                "Gemini API key not configured. Add `gemini_api_key` to your Streamlit secrets."
            )
            return

        vector_db = get_vector_db(client)

        st.title("Quotation Comparison Dashboard")
        st.write(f"Logged in as **{st.session_state.username}**")

        cols = st.columns([3, 1])
        with cols[1]:
            if st.button("Logout", key="logout_button"):
                st.session_state.logged_in = False
                st.session_state.documents = {}
                st.session_state.analysis_summary = ""
                st.session_state.chat_response = ""
                st.session_state.upload_warnings = []
                st.session_state.table_response = ""
                st.experimental_rerun()

        st.divider()

        st.subheader("📁 Upload Quotation Files")
        st.write(
            "Upload `.txt` or `.md` quotation documents. Files that are not quotations will be rejected."
        )

        save_permanent = st.checkbox("Save to permanent storage (vector database)", key="save_permanent")

        uploaded_files = st.file_uploader(
            "Select quotation files to upload",
            accept_multiple_files=True,
            type=["txt", "md"],
            key="file_uploader",
        )

        if uploaded_files and st.button("Validate and upload files", key="upload_button"):
            upload_and_validate_files(uploaded_files, client, vector_db, save_permanent)

        if st.session_state.upload_warnings:
            for warning in st.session_state.upload_warnings:
                st.warning(warning)

        render_uploaded_files(client, vector_db)

        if st.session_state.analysis_summary:
            st.divider()
            st.subheader("📊 Quotation Comparison Summary")
            st.write(st.session_state.analysis_summary)

            if st.button("Auto-generate Comparison Table", key="auto_generate_button"):
                try:
                    table_response = generate_comparison_table(
                        st.session_state.documents, client
                    )
                    st.session_state.table_response = table_response
                except Exception as exc:
                    st.error(f"Unable to generate table: {exc}")

            if "table_response" in st.session_state and st.session_state.table_response:
                st.subheader("📋 Comparison Table")
                st.markdown(st.session_state.table_response)

        render_chat_panel(client, vector_db)

    else:
        st.title("Login")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

        if submit_button:
            if check_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password. Please try again.")


if __name__ == "__main__":
    main()
