from math import sqrt
from typing import Dict, List, Tuple

from gemini_client import GeminiClient


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sqrt(sum(x * x for x in a))
    mag_b = sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


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


def build_document_embeddings(documents: Dict[str, str], client: GeminiClient) -> Dict[str, List[float]]:
    texts = list(documents.values())
    results = client.embed_texts(texts)
    return dict(zip(documents.keys(), results))


def _get_top_documents(query: str, documents: Dict[str, str], embeddings: Dict[str, List[float]], client: GeminiClient, top_k: int = 2) -> List[Tuple[str, str]]:
    query_embedding = client.embed_texts([query])
    if not query_embedding:
        return []
    query_embedding = query_embedding[0]
    scored = []
    for name, doc_embedding in embeddings.items():
        scored.append((name, _cosine_similarity(query_embedding, doc_embedding)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [(name, documents[name]) for name, _score in scored[:top_k]]


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


def answer_from_documents(query: str, documents: Dict[str, str], embeddings: Dict[str, List[float]], client: GeminiClient) -> str:
    if not documents:
        return "No quotation documents are available. Upload files first."

    top_documents = _get_top_documents(query, documents, embeddings, client, top_k=2)
    if not top_documents:
        return "Unable to retrieve relevant documents. Please try again." 

    joined_documents = "\n\n".join([f"Document: {name}\n{text[:4000]}" for name, text in top_documents])
    prompt = (
        "You are a quotation assistant. Use only the uploaded quotation documents below to answer the user's query. "
        "If the question is off-topic, reply with: 'I can only answer questions about the uploaded quotation documents.' "
        "Do not use outside knowledge or invent answers.\n\n"
        "Uploaded quotation documents:\n"
        + joined_documents
        + "\n\nUser question: "
        + query
    )
    return client.generate_text(prompt, temperature=0.2, max_output_tokens=400)
