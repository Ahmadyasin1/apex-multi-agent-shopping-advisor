import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

class VectorDBStore:
    def __init__(self, data_path="dataset.json", index_file="products.index", metadata_file="metadata.json"):
        self.data_path = data_path
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.metadata = []

    def _prepare_document_text(self, item):
        """Combine fields into a rich semantic string to embed."""
        components = [
            f"Product Name: {item.get('name', '')}",
            f"Brand: {item.get('brand', '')}",
            f"Category: {item.get('category', '')}",
            f"Price: ${item.get('price', '')}",
        ]
        if item.get('description'):
            components.append(f"Description: {item['description']}")
        if item.get('features'):
            components.append(f"Features: {', '.join(item['features'])}")
        if item.get('ideal_for'):
            components.append(f"Ideal for: {', '.join(item['ideal_for'])}")
        return " | ".join(components)

    def load_and_index(self):
        print(f"Loading data from {self.data_path}...")
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return
        
        texts = []
        for item in data:
            text = self._prepare_document_text(item)
            texts.append(text)
            self.metadata.append(item)
            
        print("Generating embeddings...")
        embeddings = self.encoder.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        # Initialize FAISS index (Inner Product for Cosine Similarity if normalized, or L2)
        # Using L2 norm index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
            
        print(f"Indexed {len(data)} products and saved to {self.index_file}.")

    def load_existing_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            return True
        return False
        
    def semantic_search(self, query: str, top_k: int = 5):
        if self.index is None:
            if not self.load_existing_index():
                print("Index not found. Please build it first.")
                return []
                
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item['_distance'] = float(dist)
                results.append(item)
                
        return results

if __name__ == "__main__":
    db = VectorDBStore(
        data_path=os.path.join(os.path.dirname(__file__), "dataset.json"),
        index_file=os.path.join(os.path.dirname(__file__), "products.index"),
        metadata_file=os.path.join(os.path.dirname(__file__), "metadata.json")
    )
    db.load_and_index()
