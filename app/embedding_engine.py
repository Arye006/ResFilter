"""
embedding_engine.py
--------------------
This is the "Resume2Vec" core: load a transformer model ONCE,
use it to turn text into vectors (embeddings), and compute how
similar two pieces of text are.

Why load the model as a singleton (only once)?
Loading a transformer model takes a few seconds and real memory.
If we reloaded it on every API request, the app would be unusably slow.
So we load it once when the server starts, and reuse it forever.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 'all-MiniLM-L6-v2' is a small, fast, free sentence-embedding model.
# It's not as powerful as a giant LLM, but for resume/JD semantic matching
# it punches well above its weight and runs fine on a laptop CPU.
_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # will hold the loaded model after first use


def get_model() -> SentenceTransformer:
    """
    Lazy-load the model: load it the FIRST time it's needed,
    then reuse the same instance for every later call.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {_MODEL_NAME} ...")
        _model = SentenceTransformer(_MODEL_NAME)
        print("Model loaded.")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into a list of embedding vectors.
    Batching like this (passing a list, not one string at a time) is
    much faster than calling the model once per resume.
    """
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def compute_similarity(jd_embedding: np.ndarray, resume_embeddings: np.ndarray) -> list[float]:
    """
    Compare ONE job description embedding against MANY resume embeddings.
    Returns a plain Python list of similarity scores, one per resume,
    each between -1 (opposite meaning) and 1 (identical meaning).
    In practice for resumes/JDs you'll mostly see scores between 0 and 1.
    """
    jd_embedding = jd_embedding.reshape(1, -1)  # reshape to (1, vector_size)
    similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]
    return similarities.tolist()
