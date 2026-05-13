import requests
from typing import Dict, List


class GeminiClient:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta2"

    def __init__(self, api_key: str, model: str = "gemini-1.0", embed_model: str = "gemini-embedding-1.0"):
        self.api_key = api_key
        self.model = model
        self.embed_model = embed_model

    def _post(self, endpoint: str, payload: Dict) -> Dict:
        url = f"{self.BASE_URL}/models/{endpoint}"
        response = requests.post(url, params={"key": self.api_key}, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def generate_text(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 512) -> str:
        payload = {
            "prompt": {"text": prompt},
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        result = self._post(f"{self.model}:generateText", payload)
        return result["candidates"][0].get("content", "").strip()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = {"input": texts}
        result = self._post(f"{self.embed_model}:embedText", payload)
        embeddings = [item["embedding"] for item in result.get("embeddings", [])]
        return embeddings
