import json
import math
from pathlib import Path
from typing import Dict, List

from gemini_client import GeminiClient


class VectorDB:
    def __init__(self, client: GeminiClient, persist_directory: str = "vector_db"):
        self.client = client
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.data_path = self.persist_directory / "vector_store.json"
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, Dict]:
        if self.data_path.exists():
            try:
                return json.loads(self.data_path.read_text(encoding="utf-8"))
            except Exception:
                return {"chunks": []}
        return {"chunks": []}

    def _save_state(self) -> None:
        self.data_path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add_document(self, source_name: str, text: str) -> None:
        chunks = self._chunk_text(text)
        if not chunks:
            return

        embeddings = self.client.embed_texts(chunks)
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self._state["chunks"].append(
                {
                    "id": f"{source_name}__{idx}",
                    "source": source_name,
                    "text": chunk,
                    "embedding": embedding,
                }
            )
        self._save_state()

    def delete_document(self, source_name: str) -> None:
        self._state["chunks"] = [
            chunk for chunk in self._state["chunks"] if chunk.get("source") != source_name
        ]
        self._save_state()

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, str]]:
        query_embedding = self.client.embed_texts([query_text])
        if not query_embedding:
            return []

        query_embedding = query_embedding[0]
        scored = []
        for chunk in self._state.get("chunks", []):
            similarity = self._cosine_similarity(query_embedding, chunk.get("embedding", []))
            scored.append((similarity, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_results = [item[1] for item in scored[:top_k] if item[0] > 0]

        return [
            {"text": chunk["text"], "source": chunk["source"]}
            for chunk in top_results
        ]

    def list_sources(self) -> List[str]:
        sources = sorted({chunk.get("source") for chunk in self._state.get("chunks", []) if chunk.get("source")})
        return sources
