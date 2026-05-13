from typing import Dict, List

from gemini_client import GeminiClient


def validate_quotation_document(text: str, client: GeminiClient) -> bool:
    prompt = (
        "You are a quotation document validator. "
        "The user has uploaded a document that may be a vendor quotation, proposal, or estimate. "
        "Respond with only YES or NO. "
        "Return YES if the document is a quotation or estimate provided by a vendor or supplier. "
        "Return NO if the document is not a quotation, is unrelated, or is not a quote for goods or services.\n\n"
        "Document:\n" + text[:4000]
    )
    response = client.generate_text(prompt, temperature=0.0, max_output_tokens=40)
    normalized = response.strip().lower()
    return normalized.startswith("yes")


def compare_quotes(documents: Dict[str, str], client: GeminiClient) -> str:
    if not documents:
        return "Upload quotation documents first to generate a comparison summary."

    joined_documents = "\n\n".join(
        [f"Document: {name}\n{text[:4000]}" for name, text in documents.items()]
    )
    prompt = (
        "You are a quotation comparison assistant. "
        "Compare the uploaded quotation documents and provide a pros and cons summary for each file. "
        "Identify strengths, weaknesses, risks, pricing considerations, and notable terms. "
        "Use only the content from the uploaded documents and do not invent information.\n\n"
        "Uploaded quotations:\n"
        + joined_documents
    )
    return client.generate_text(prompt, temperature=0.2, max_output_tokens=400)


def generate_comparison_table(documents: Dict[str, str], client: GeminiClient) -> str:
    if not documents:
        return "Upload quotation documents first to generate a comparison table."

    joined_documents = "\n\n".join(
        [f"Document: {name}\n{text[:4000]}" for name, text in documents.items()]
    )
    prompt = (
        "You are a quotation comparison assistant. "
        "Create a comparison table for the uploaded quotation documents. "
        "The table should have parameters as rows and vendor names as columns. "
        "Include parameters like: Pricing, Terms, Delivery Time, Quality, Support, Risks, etc. "
        "For each parameter and vendor, provide specific information from the documents. "
        "Output the table in Markdown format with | separators. "
        "Use only the content from the uploaded documents and do not invent information.\n\n"
        "Uploaded quotations:\n"
        + joined_documents
    )
    return client.generate_text(prompt, temperature=0.2, max_output_tokens=600)


def answer_from_vector_db(query: str, vector_db, client: GeminiClient, top_k: int = 3) -> str:
    if not vector_db or not vector_db.list_sources():
        return "No quotation documents are available. Upload files first."

    relevant_chunks = vector_db.query(query, top_k=top_k)
    if not relevant_chunks:
        return "I can only answer questions about the uploaded quotation documents."

    joined_chunks = "\n\n".join(
        [f"Source: {chunk['source']}\n{chunk['text'][:1200]}" for chunk in relevant_chunks]
    )
    prompt = (
        "You are a quotation assistant. Answer using only the uploaded quotation document chunks below. "
        "If the question is off-topic or unrelated to the uploaded quotations, reply with: 'I can only answer questions about the uploaded quotation documents.' "
        "Do not use outside knowledge or invent answers.\n\n"
        "Relevant quotation chunks:\n"
        + joined_chunks
        + "\n\nUser question: "
        + query
    )
    return client.generate_text(prompt, temperature=0.2, max_output_tokens=400)
