import logging
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_pinecone_index = None


def get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.pinecone_api_key)

        existing = [i.name for i in pc.list_indexes()]
        if settings.pinecone_index_name not in existing:
            logger.info(f"Creating Pinecone index: {settings.pinecone_index_name}")
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.pinecone_environment,
                ),
            )

        _pinecone_index = pc.Index(settings.pinecone_index_name)
        logger.info(f"Connected to Pinecone index: {settings.pinecone_index_name}")

    return _pinecone_index


def upsert_chunks(chunks: List[dict], embeddings: List[List[float]]) -> int:
    """Store chunks with embeddings in Pinecone."""
    index = get_index()

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vector_id = f"{chunk['document_id']}_{chunk['chunk_index']}"
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": chunk["text"][:1000],
                "page": chunk.get("page", 0),
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
            },
        })

    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)
        logger.info(f"Upserted batch {i // BATCH_SIZE + 1}: {len(batch)} vectors")

    return len(vectors)


def similarity_search(
    query_embedding: List[float],
    top_k: int = 5,
    document_id: Optional[str] = None,
) -> List[dict]:
    """Retrieve top-k similar chunks, optionally filtered by document."""
    index = get_index()

    filter_dict = {}
    if document_id:
        filter_dict["document_id"] = {"$eq": document_id}

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None,
    )

    matches = []
    for match in results.matches:
        if match.score < 0.2:
            continue
        matches.append({
            "text": match.metadata.get("text", ""),
            "page": match.metadata.get("page"),
            "score": round(match.score, 4),
            "document_id": match.metadata.get("document_id"),
        })

    return matches