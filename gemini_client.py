import requests
import base64
from typing import Dict, List

class GeminiClient:
    BASE_URL = "https://generativelanguage.googleapis.com/v1"

    # Defaulting to Pro as confirmed available for your key
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite", embed_model: str = "text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self.embed_model = embed_model

    def _post(self, endpoint: str, payload: Dict) -> Dict:
        # Crucial: This already includes /models/
        url = f"{self.BASE_URL}/models/{endpoint}"
        response = requests.post(
            url, 
            params={"key": self.api_key}, 
            json=payload, 
            timeout=60 
        )
        
        if response.status_code != 200:
            error_msg = response.json().get("error", {}).get("message", "Unknown API Error")
            raise Exception(f"Gemini API Error: {error_msg}")
        
        return response.json()

    def generate_text(self, prompt: str, temperature: float = 0.1, max_output_tokens: int = 1024) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": 0.95,
            }
        }
        try:
            # FIX: Removed 'models/' prefix because _post() adds it
            endpoint = f"{self.model}:generateContent"
            result = self._post(endpoint, payload)
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_content_from_bytes(self, file_bytes: bytes, mime_type: str, prompt: str, temperature: float = 0.0) -> str:
        encoded_file = base64.b64encode(file_bytes).decode("utf-8")
        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": encoded_file}},
                    {"text": prompt}
                ]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1024
            }
        }
        try:
            # FIX: Removed 'models/' prefix here as well
            endpoint = f"{self.model}:generateContent"
            result = self._post(endpoint, payload)
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            return f"Multimodal analysis failed: {str(e)}"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        embeddings = []
        try:
            for text in texts:
                payload = {
                    "model": f"models/{self.embed_model}",
                    "content": {"parts": [{"text": text}]}
                }
                # FIX: Pass only the embedding model ID and method
                result = self._post(f"{self.embed_model}:embedContent", payload)
                embeddings.append(result["embedding"]["values"])
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
        
    def answer_from_full_context(query: str, documents: Dict[str, str], client: GeminiClient) -> str:
        """
        Directly uses the 1M token limit of Gemini 2.5 Pro to answer queries 
        without needing a vector database.
        """
        if not documents:
            return "No documents available to discuss."

        # Join all documents into one massive context block
        full_context = "\n\n".join([f"Document: {name}\n{text}" for name, text in documents.items()])
        
        prompt = (
            "SYSTEM: You are a secure technical auditor. Answer the user question ONLY using "
            "the context provided below. If the answer is not there, say so.\n\n"
            f"CONTEXT:\n{full_context}\n\n"
            f"USER QUESTION: {query}"
        )
        
        # Gemini 2.5 Pro can handle this effortlessly
        return client.generate_text(prompt, temperature=0.1)
    

    def generate_content_from_multiple_pdfs(self, file_data_list: List[Dict], prompt: str) -> str:
        parts = []
        for file_info in file_data_list:
            encoded = base64.b64encode(file_info["bytes"]).decode("utf-8")
            parts.append({"inline_data": {"mime_type": file_info["mime_type"], "data": encoded}})

        parts.append({"text": prompt})
        payload = {
                    "contents": [{"parts": parts}], 
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 2048  # Increased for complex technical evaluation
                    }
                }

        result = self._post(f"{self.model}:generateContent", payload)
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()