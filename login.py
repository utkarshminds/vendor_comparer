import re
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
    validate_rfq_document_multimodal,         # Update this
    extract_rfq_requirements, # Add this new import
    extract_requirements_expansion_step
)
from storage import (
    allowed_file_type,
    create_uploads_directory,
    delete_uploaded_file,
    extract_uploaded_text,
    save_uploaded_file,
)
from vector_db import VectorDB

def render_multi_color_table(md_text: str) -> None:
    """
    Parses a Markdown table and renders it as an HTML table with distinct 
    colors for the RFQ baseline column vs each unique vendor column.
    """
    if "|" not in md_text:
        st.markdown(md_text)
        return

    lines = md_text.split("\n")
    table_lines = [l.strip() for l in lines if l.strip().startswith("|")]

    if len(table_lines) < 3:
        st.markdown(md_text)
        return

    try:
        # Extract headers and body rows
        headers = [h.strip() for h in table_lines[0].split("|")[1:-1]]
        rows = []
        for l in table_lines[2:]:
            if l.strip().startswith("|"):
                rows.append([c.strip() for c in l.split("|")[1:-1]])

        # Distinct color palette per column index:
        # Col 0 (RFQ/Parameters) | Col 1 (Vendor A) | Col 2 (Vendor B) | Col 3 (Vendor C)
        cell_bgs = ["#eaf2f8", "#e8f8f5", "#fef9e7", "#f5eef8", "#fdf2e9", "#f4f6f7"]
        header_bgs = ["#2e4053", "#117a65", "#b7950b", "#6c3483", "#a04000", "#566573"]

        html_output = (
            "<div style='overflow-x:auto; margin-top: 15px;'>"
            "<table style='width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px;'>"
        )

        # Build Headers
        html_output += "<tr>"
        for i, h in enumerate(headers):
            bg = header_bgs[i % len(header_bgs)]
            html_output += f"<th style='background-color: {bg}; border: 1px solid #bdc3c7; padding: 12px; text-align: left; color: white; font-weight: bold;'>{h}</th>"
        html_output += "</tr>"

        # Build Dynamic Rows
        for row in rows:
            if not any(row): continue
            html_output += "<tr>"
            for i, val in enumerate(row):
                bg = cell_bgs[i % len(cell_bgs)]
                # Preserve simple markdown bolding **text** inside HTML cells
                clean_val = val.replace("**", "<strong>").replace("**", "</strong>")
                html_output += f"<td style='background-color: {bg}; border: 1px solid #d5dbdb; padding: 12px; text-align: left; color: #2c3e50;'>{clean_val}</td>"
            html_output += "</tr>"

        html_output += "</table></div>"
        st.markdown(html_output, unsafe_allow_html=True)
    except Exception:
        # Seamless fallback to default presentation if parsing drops out
        st.markdown(md_text)
        
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
    if "rfq_requirements" not in st.session_state:
        st.session_state.rfq_requirements = []  # Changed from [] to an empty string
    if "chat_query" not in st.session_state:
        st.session_state.chat_query = ""
    if "selected_requirement" not in st.session_state:
        st.session_state.selected_requirement = ""

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

def upload_and_validate_files(uploaded_files, client: GeminiClient, vector_db: VectorDB) -> None:
    new_documents = {}
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        # 1. Multimodal validation (No text extraction needed)
        if validate_quotation_document_multimodal(file_bytes, "application/pdf", client):
            # 2. Store in memory for Gemini 3.1 Long Context
            new_documents[uploaded_file.name] = {
                "bytes": file_bytes,
                "mime_type": "application/pdf"
            }
            # 3. Save to temp disk so the UI shows the correct KB
            save_uploaded_file(uploaded_file) 

            # 4. RAG IMPLEMENTATION: Extract text and add to Vector DB
            extracted_text = extract_uploaded_text(uploaded_file)
            if extracted_text:
                vector_db.add_document(uploaded_file.name, extracted_text)

        else:
            st.error(f"Validation failed for {uploaded_file.name}")
            
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
    # Use the RAG function instead of the multimodal one
    
    
    st.session_state.chat_query = st.text_input("Ask a technical question about the bids:")
    
    if st.button("Submit"):
        if not st.session_state.chat_query:
            st.warning("Please enter a question.")
            return
            
        with st.spinner("Searching vector database..."):
            response = answer_from_vector_db(
                query=st.session_state.chat_query,
                vector_db=vector_db,
                client=client
            )
            st.session_state.chat_response = response
            
    if st.session_state.chat_response:
        st.markdown("### Answer")
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
                
                # CONDITIONAL RENDERING: Only show extract button once validated baseline exists but list is empty
                if not st.session_state.rfq_requirements:
                    st.info("Validation complete. Click below to pull all itemized engineering requirements.")
                    
                    if st.button("🔧 Extract Technical Requirements", type="primary"):
                        import time
                        
                        total_passes = 5
                        est_seconds_per_pass = 5 
                        requirements_list = []   
                        
                        progress_bar = st.progress(0, text="Initializing file sequence...")
                        
                        for i in range(1, total_passes + 1):
                            time_left = (total_passes - i + 1) * est_seconds_per_pass
                            percent_complete = int(((i - 1) / total_passes) * 100)
                            
                            progress_bar.progress(
                                percent_complete, 
                                text=f"⏳ Running Extraction Loop {i}/{total_passes}... [~{time_left}s remaining]"
                            )
                            
                            requirements_list = extract_requirements_expansion_step(
                                st.session_state.rfq_raw_bytes,
                                requirements_list,
                                i,
                                client
                            )
                        
                        progress_bar.progress(100, text="✅ Technical requirements completely captured!")
                        time.sleep(1)
                        
                        # Set to state as a strict verified Python list
                        st.session_state.rfq_requirements = list(requirements_list)
                        st.rerun()
                else:
                    st.markdown("### 📊 Extracted Technical Requirements Matrix")
                    
                    raw_reqs = st.session_state.rfq_requirements
                    final_list = [raw_reqs] if isinstance(raw_reqs, str) else list(raw_reqs)
                    
                    html_table = (
                        "<div style='overflow-x:auto; width: 100%; margin-top: 10px;'>"
                        "<table style='width: 100%; min-width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px;'>"
                        "<tr style='background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;'>"
                        "<th style='width: 8%; padding: 12px; text-align: left; font-weight: 600; color: #495057;'>Sr. No.</th>"
                        "<th style='width: 92%; padding: 12px; text-align: left; font-weight: 600; color: #495057;'>Technical Requirement Specification</th>"
                        "</tr>"
                    )
                    
                    for idx, item in enumerate(final_list):
                        text = str(item).strip()
                        
                        # CRITICAL: Boundary-only cleanup engine applied before display rendering.
                        # Internal quote expressions or escaped slashes remain intact.
                        text = re.sub(r'^[\"\\\'\s\u201c\u201d,]+', '', text)
                        clean_item = re.sub(r'[\"\\\'\s\u201c\u201d,]+$', '', text)
                        
                        bg_color = "#ffffff" if idx % 2 == 0 else "#fdfdfd"
                        
                        html_table += (
                            f"<tr style='background-color: {bg_color}; border-bottom: 1px solid #e9ecef;'>\n"
                            f"<td style='padding: 14px 12px; color: #6c757d; vertical-align: top;'>{idx + 1}</td>\n"
                            f"<td style='padding: 14px 12px; color: #212529; line-height: 1.5; text-align: justify;'>{clean_item}</td>\n"
                            f"</tr>"
                        )
                        
                    html_table += "</table></div>"
                    st.markdown(html_table, unsafe_allow_html=True)

                if cols_rfq[1].button("Reset RFQ", type="secondary"):
                    st.session_state.rfq_text = ""
                    st.session_state.rfq_filename = ""
                    st.session_state.rfq_requirements = [] 
                    if "rfq_raw_bytes" in st.session_state:
                        del st.session_state.rfq_raw_bytes
                    st.rerun()

            st.divider()

            # 2. VENDOR BIDS SECTION
            st.subheader("2. Technical Vendor Bids")
            if not st.session_state.rfq_filename:
                st.warning("Please upload a baseline RFQ above before processing vendor bids.")
            else:
                uploaded_bids = st.file_uploader(
                    "Upload Technical Bids", 
                    accept_multiple_files=True, 
                    type=["pdf"]
                )
                
                if uploaded_bids and st.button("Validate & Process Bids"):
                    upload_and_validate_files(uploaded_bids, client, vector_db)
                
                if st.session_state.upload_warnings:
                    for w in st.session_state.upload_warnings:
                        st.warning(w)

                render_uploaded_files(client, vector_db)

                # 3. ANALYSIS SECTION
                if st.session_state.documents:
                    st.divider()
                    st.subheader("3. Automated Technical Evaluation")
                    if st.button("Run Full Comparison Analysis", type="primary"):
                        if not st.session_state.get("rfq_raw_bytes"):
                            st.error("Please upload the Request for Quotation (RFQ) first.")
                        elif not st.session_state.documents:
                            st.error("Please upload at least one technical bid to evaluate.")
                        else:
                            with st.spinner("Analyzing bids against technical norms..."):
                                from quote_system import evaluate_bids_multimodal
                                summary = evaluate_bids_multimodal(
                                    st.session_state.rfq_raw_bytes, 
                                    st.session_state.documents, 
                                    client
                                )
                                st.session_state.analysis_summary = summary
                                st.rerun()
                    
                    if st.session_state.analysis_summary:
                        st.markdown("### Evaluation Summary Matrix")
                        render_multi_color_table(st.session_state.analysis_summary)

        # --- TAB 2: SECURE CHAT ---
        with tab_chat:
            if not st.session_state.documents:
                st.info("The chat interface will activate once technical bids are uploaded and validated.")
            else:
                st.subheader("💬 Technical Query Interface")
                st.caption("Answers are strictly limited to the content of validated technical bids.")
                # Pass vector_db as the second argument
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
