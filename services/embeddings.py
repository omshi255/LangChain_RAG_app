import logging
from typing import List
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global embedding model instance
_embedding_model = None


def get_embedding_model():
    """Lazy-load FastEmbed model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from fastembed import TextEmbedding

            logger.info("Loading FastEmbed embedding model...")
            _embedding_model = TextEmbedding()  # default model (BAAI/bge-small-en)
            logger.info("FastEmbed model loaded successfully")

        except ImportError:
            logger.error("fastembed not installed. Install with: pip install fastembed")
            raise

    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate semantic embeddings."""
    if not texts:
        return []

    model = get_embedding_model()

    # Clean texts
    normalized_texts = [text.strip() for text in texts if text.strip()]

    if not normalized_texts:
        logger.warning("All texts empty after normalization")
        return [[0.0] * 384 for _ in texts]

    try:
        embeddings = list(model.embed(normalized_texts))
        embeddings = [emb.tolist() for emb in embeddings]

        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise


def embed_query(text: str) -> List[float]:
    """Generate embedding for a single query."""
    result = embed_texts([text])
    return result[0] if result else []