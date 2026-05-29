import http.client
import json
import urllib.parse
from typing import Dict, Any, Optional, List
import sys
import os

# Append parent dir so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config

class RapidAPIClient:
    """
    Generalized and secure RapidAPI client with timeout and robust error handling.
    """
    def __init__(self):
        # We fetch the key dynamically so it's always up to date
        self.api_key = Config.RAPIDAPI_KEY
        self.timeout = 15  # Fallback timeout

    def _safe_request(self, host: str, endpoint: str, method: str = "GET", payload: Optional[str] = None) -> Dict[str, Any]:
        """Core secure request method with exception handling."""
        if not self.api_key or self.api_key == "MISSING_KEY":
            print(f"[Security Warning] Missing RAPIDAPI_KEY. Skipping {host} request.")
            return {}

        conn = None
        try:
            conn = http.client.HTTPSConnection(host, timeout=self.timeout)
            headers = {
                'x-rapidapi-key': self.api_key.strip(),
                'x-rapidapi-host': host,
                'Content-Type': "application/json"
            }
            # Ensure payload is encoded to utf-8 if present
            encoded_payload = payload.encode('utf-8') if payload else None
            conn.request(method, endpoint, body=encoded_payload, headers=headers)
            res = conn.getresponse()
            data = res.read()
            raw_text = str(data.decode("utf-8"))
            
            if res.status >= 400:
                short_text = raw_text[0:199] if len(raw_text) > 200 else raw_text
                print(f"[RapidAPI Error] {host}{endpoint} returned status {res.status}: {short_text}")
                return {"error": True, "status": res.status, "message": short_text}

            # Prevent empty decode errors
            if not raw_text.strip():
                return {}

            return dict(json.loads(raw_text))
        except json.JSONDecodeError as e:
            print(f"[RapidAPI Parse Error] Failed to parse JSON from {host}: {e}")
            return {"error": True, "message": "Invalid JSON response"}
        except Exception as e:
            print(f"[RapidAPI Connection Error] {host}{endpoint} - {str(e)}")
            return {"error": True, "message": str(e)}
        finally:
            if conn:
                conn.close()

    def get_realtime_product_search(self, query: str, country: str = "us") -> List[Dict[str, Any]]:
        """Generic endpoint assumption for product search."""
        host = "real-time-product-search.p.rapidapi.com"
        safe_query = urllib.parse.quote(query)
        # Assuming standard /search endpoint for this API based on the pattern
        endpoint = f"/search?q={safe_query}&country={country}&language=en"
        response = self._safe_request(host, endpoint)
        
        # Parse standard response formats
        if isinstance(response, dict) and "data" in response:
            return response.get("data", [])
        elif isinstance(response, list):
            return response
        return []

    def get_amazon_offers(self, asin: str) -> Dict[str, Any]:
        """Provides Amazon offers securely."""
        host = "real-time-amazon-data.p.rapidapi.com"
        # Validate ASIN vaguely to prevent malformed requests
        if not asin or not asin.isalnum():
            return {"error": True, "message": "Invalid ASIN"}
        endpoint = f"/product-offers?asin={asin}&country=US&limit=10&page=1"
        return self._safe_request(host, endpoint)

    def get_ebay_product(self, product_id: str) -> Dict[str, Any]:
        """Fetches eBay product securely."""
        host = "ebay32.p.rapidapi.com"
        if not product_id.isdigit():
            return {"error": True, "message": "Invalid Product ID shape"}
        endpoint = f"/product/{product_id}?country=germany&country_code=de"
        return self._safe_request(host, endpoint)

    def ask_claude_sonnet(self, prompt: str) -> str:
        """Invokes Claude 3.7 Sonnet as an alternative reasoning engine."""
        host = "claude-3-7-sonnet.p.rapidapi.com"
        endpoint = "/"
        
        # Sanitize and prepare payload
        safe_content = prompt.replace("\"", "\\\"").replace("\n", "\\n")
        payload = f'{{"model":"claude-3-7-sonnet","messages":[{{"role":"user","content":"{safe_content}"}}]}}'
        
        response = self._safe_request(host, endpoint, method="POST", payload=payload)
        
        # Extract Claude's response (assuming standard OpenAI/Claude API layout from RapidAPI Wrapper)
        try:
            if "choices" in response:
                return response["choices"][0]["message"]["content"]
            elif "content" in response:
                if isinstance(response["content"], list):
                    return response["content"][0]["text"]
                return str(response["content"])
        except (KeyError, IndexError):
            pass
        
        if response.get("error"):
            return f"Claude API Error: {response.get('message')}"
            
        return json.dumps(response)

# Singleton instance
rapidapi_client = RapidAPIClient()
