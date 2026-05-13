from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from gemini_client import GeminiClient


class VectorDB:
    COLLECTION_NAME = "quotations"

    def __init__(self, client: GeminiClient, persist_directory: str = "vector_db"):
        self.client = client
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.chromadb_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.persist_directory),
            )
        )
        self.collection = self.chromadb_client.get_or_create_collection(
            name=self.COLLECTION_NAME
        )

    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start = end - overlap
        return chunks

    def add_document(self, source_name: str, text: str) -> None:
        chunks = self._chunk_text(text)
        if not chunks:
            return

        embeddings = self.client.embed_texts(chunks)
        chunk_ids = [f"{source_name}__{idx}" for idx in range(len(chunks))]
        metadatas = [
            {"source": source_name, "chunk_index": idx}
            for idx in range(len(chunks))
        ]

        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def delete_document(self, source_name: str) -> None:
        self.collection.delete(where={"source": source_name})

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, str]]:
        query_embedding = self.client.embed_texts([query_text])
        if not query_embedding:
            return []

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        if not results or not results.get("documents"):
            return []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [
            {"text": chunk, "source": metadata.get("source", "unknown")}
            for chunk, metadata in zip(documents, metadatas)
        ]

    def list_sources(self) -> List[str]:
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for metadata in results.get("metadatas", []):
            for item in metadata:
                source = item.get("source")
                if source:
                    sources.add(source)
        return sorted(sources)
