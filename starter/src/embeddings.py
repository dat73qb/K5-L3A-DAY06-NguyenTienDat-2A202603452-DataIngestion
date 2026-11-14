"""A real embedding provider (local Sentence Transformers, multilingual).

Vectors are L2-normalized so pgvector's cosine distance operator ``<=>``
returns the equivalent of 1 - cosine_similarity, and the reported score is
cosine similarity in [0, 1].
"""
from typing import Protocol

from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Adapter over SentenceTransformer.

    Defaults to the E5 query/passage prefix convention, which is required for
    strong cross-lingual retrieval with ``intfloat/multilingual-e5-base``.
    Pass empty prefixes for non-E5 models (e.g. MiniLM).
    """

    def __init__(self, model_name: str,
                 query_prefix: str = "query: ",
                 passage_prefix: str = "passage: ") -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [self.passage_prefix + t for t in texts]
        vectors = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, question: str) -> list[float]:
        vector = self._model.encode(
            [self.query_prefix + question],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return list(map(float, vector))
