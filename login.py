from email.mime import text
from http import client
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
    evaluate_bids_against_rfq, # Use the new evaluation logic
    generate_comparison_table,
    validate_quotation_document,
    validate_rfq_document,        # Add this for the new workflow
    validate_quotation_document_multimodal,  # Update this
    validate_rfq_document_multimodal         # Update this
)
from storage import (
    allowed_file_type,
    create_uploads_directory,
    delete_uploaded_file,
    extract_uploaded_text,
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
    if "rfq_text" not in st.session_state:
        st.session_state.rfq_text = ""
    if "rfq_filename" not in st.session_state:
        st.session_state.rfq_filename = ""

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

def upload_and_validate_files(uploaded_files, client: GeminiClient) -> None:
    new_documents = {}
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        if validate_quotation_document_multimodal(file_bytes, "application/pdf", client):
            # Store bytes and mime_type for multimodal analysis
            new_documents[uploaded_file.name] = {
                "bytes": file_bytes,
                "mime_type": "application/pdf"
            }
    st.session_state.documents.update(new_documents)

def delete_file(filename: str, client: GeminiClient, vector_db: VectorDB) -> None:
    if delete_uploaded_file(filename):
        st.session_state.documents.pop(filename, None)
        vector_db.delete_document(filename)
        
        # Clear the old evaluations so the user can run a fresh one
        st.session_state.analysis_summary = ""
        st.session_state.table_response = ""
        
        st.success(f"Deleted {filename} successfully.")


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
    st.set_page_config(page_title="Secure Bid Evaluator", layout="wide")
    init_session_state()

    if st.session_state.logged_in:
        client = get_gemini_client()
        if not client:
            st.error("Gemini API key not configured. Check your secure environment variables.")
            return

        vector_db = get_vector_db(client)

        # Header with Logout
        col_title, col_logout = st.columns([4, 1])
        with col_title:
            st.title("🛡️ Secure Technical Bid Evaluation")
            st.caption(f"Authenticated as: {st.session_state.username}")
        with col_logout:
            if st.button("Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.divider()

        # Implementation of the Tabbed Interface
        tab_eval, tab_chat = st.tabs(["⚖️ Evaluation Dashboard", "💬 Secure Technical Chat"])

        # --- TAB 1: EVALUATION DASHBOARD ---
        with tab_eval:
            # 1. RFQ/RFO BASELINE SECTION
            st.subheader("1. Request for Quotation (RFQ) Baseline")
            
            if not st.session_state.get("rfq_filename"):
                st.info("Upload the project requirements (RFQ) first to establish evaluation norms.")
                rfq_file = st.file_uploader(
                    "Upload RFQ/RFO Document", 
                    type=["pdf"], 
                    key="rfq_uploader"
                )
                # Change from extracting text to storing raw bytes
                if rfq_file and st.button("Set Baseline"):
                    file_bytes = rfq_file.getvalue()
                    mime_type = "application/pdf"
                    with st.spinner("Analyzing baseline..."):
                        if validate_rfq_document_multimodal(file_bytes, mime_type, client):
                            st.session_state.rfq_raw_bytes = file_bytes  # Store raw bytes
                            st.session_state.rfq_filename = rfq_file.name
                            st.success(f"Baseline set: {rfq_file.name}")
                            st.rerun()
                        else:
                            st.error("Rejected: The document does not meet the technical criteria for an RFQ/MR.")
            else:
                cols_rfq = st.columns([3, 1])
                cols_rfq[0].success(f"**Active Baseline:** {st.session_state.rfq_filename}")
                if cols_rfq[1].button("Reset RFQ", type="secondary"):
                    st.session_state.rfq_text = ""
                    st.session_state.rfq_filename = ""
                    st.rerun()

            st.divider()

            # 2. VENDOR BIDS SECTION
            st.subheader("2. Technical Vendor Bids")
            if not st.session_state.rfq_filename:
                st.warning("Please upload a baseline RFQ above before processing vendor bids.")
            else:
                save_perm = st.checkbox("Commit to Secure Vector Vault", value=True)
                uploaded_bids = st.file_uploader(
                    "Upload Technical Bids", 
                    accept_multiple_files=True, 
                    type=["pdf"]
                )
                
                if uploaded_bids and st.button("Validate & Process Bids"):
                    upload_and_validate_files(uploaded_bids, client, vector_db, save_perm)
                
                if st.session_state.upload_warnings:
                    for w in st.session_state.upload_warnings:
                        st.warning(w)

                render_uploaded_files(client, vector_db)

                # 3. ANALYSIS SECTION
                if st.session_state.documents:
                    st.divider()
                    st.subheader("3. Automated Technical Evaluation")
                    if st.button("Run Full Comparison Analysis", type="primary"):
                        with st.spinner("Analyzing bids against technical norms..."):
                            from quote_system import evaluate_bids_against_rfq
                            summary = evaluate_bids_against_rfq(
                                st.session_state.rfq_text, 
                                st.session_state.documents, 
                                client
                            )
                            st.session_state.analysis_summary = summary
                    
                    if st.session_state.analysis_summary:
                        st.markdown("### Evaluation Summary")
                        st.write(st.session_state.analysis_summary)

        # --- TAB 2: SECURE CHAT ---
        with tab_chat:
            if not st.session_state.documents:
                st.info("The chat interface will activate once technical bids are uploaded and validated.")
            else:
                st.subheader("💬 Technical Query Interface")
                st.caption("Answers are strictly limited to the content of validated technical bids.")
                render_chat_panel(client, vector_db)

    else:
        # Standard Login Form
        st.title("🔐 Corporate Login")
        with st.form("login_form"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Access System"):
                if check_credentials(user, pw):
                    st.session_state.logged_in = True
                    st.session_state.username = user
                    st.rerun()
                else:
                    st.error("Authentication failed. Please check credentials.")

if __name__ == "__main__":
    main()
