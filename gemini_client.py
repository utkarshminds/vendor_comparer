import requests
from typing import Dict, List

class GeminiClient:
    BASE_URL = "https://generativelanguage.googleapis.com/v1"

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro", embed_model: str = "text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self.embed_model = embed_model

    def _post(self, endpoint: str, payload: Dict) -> Dict:
        url = f"{self.BASE_URL}/models/{endpoint}"
        # Security: Pass the API key in headers if supported, or ensure params are not logged
        response = requests.post(
            url, 
            params={"key": self.api_key}, 
            json=payload, 
            timeout=60 # Increased timeout for large technical documents
        )
        
        if response.status_code != 200:
            # Masking potential sensitive info in error logs
            error_msg = response.json().get("error", {}).get("message", "Unknown API Error")
            raise Exception(f"Gemini API Error: {error_msg}")
        
        return response.json()
    
    def list_available_models(api_key: str):
        """
        Lists all models available to the provided Gemini API key.
        Useful for identifying the correct model string (e.g., 'gemini-1.5-flash').
        """
        url = "https://generativelanguage.googleapis.com/v1/models"
        response = requests.get(url, params={"key": api_key})
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("--- Available Models for your API Key ---")
            for m in models:
                # Filter for models that support text generation
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    print(f"Model ID: {m['name']} | Description: {m['displayName']}")
        else:
            print(f"Failed to list models: {response.status_code} - {response.text}")



    def generate_text(self, prompt: str, temperature: float = 0.1, max_output_tokens: int = 1024) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": 0.95, # Added for better response quality
            }
        }
        
        try:
            result = self._post(f"{self.model}:generateContent", payload)
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            # Log this internally, but return a clean error to the UI
            return f"Error generating response: {str(e)}"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # Batching logic: Google allows up to 100 texts per batchEmbedContents call
        # For simplicity in this secure version, we still process one by one but 
        # with enhanced error handling.
        embeddings = []
        try:
            for text in texts:
                payload = {
                    "model": f"models/{self.embed_model}",
                    "content": {"parts": [{"text": text}]}
                }
                result = self._post(f"{self.embed_model}:embedContent", payload)
                embeddings.append(result["embedding"]["values"])
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")