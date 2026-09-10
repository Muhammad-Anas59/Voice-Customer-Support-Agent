"""
embedder.py

Job: Take the text chunks from document_loader.py, convert each one into an
embedding (a vector of numbers representing its meaning) using Gemini's
embedding model, then store all embeddings in a FAISS index so we can later
search "which chunks are most similar to this customer question."
"""

import os
import numpy as np
import faiss
from google import genai
from google.genai.types import EmbedContentConfig
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"


def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    """Sends one piece of text to Gemini and gets back its embedding vector.
    task_type differs for documents vs queries - Gemini optimizes each differently."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=EmbedContentConfig(task_type=task_type)
    )
    return result.embeddings[0].values


def build_faiss_index(chunks):
    """Takes the list of chunks (from document_loader), embeds every one,
    and builds a FAISS index for fast similarity search.
    Returns: (faiss_index, chunks_with_embeddings)"""
    print(f"Embedding {len(chunks)} chunks...")
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk["text"], task_type="RETRIEVAL_DOCUMENT")
        vectors.append(vector)
        print(f"  embedded chunk {i+1}/{len(chunks)}")

    vectors_np = np.array(vectors).astype("float32")
    dimension = vectors_np.shape[1]

    # IndexFlatL2 = simple, exact similarity search (fine for our small dataset size)
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    return index, chunks


def save_index(index, chunks, folder="app/data"):
    """Saves the FAISS index and the chunk metadata to disk, so we don't
    have to re-embed everything every time we run the app."""
    os.makedirs(folder, exist_ok=True)
    faiss.write_index(index, os.path.join(folder, "policy_index.faiss"))

    import json
    with open(os.path.join(folder, "chunks_metadata.json"), "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved index and metadata to {folder}/")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from document_loader import load_and_chunk_all

    chunks = load_and_chunk_all("policy_docs")
    index, chunks_with_meta = build_faiss_index(chunks)
    save_index(index, chunks_with_meta)
