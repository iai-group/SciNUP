"""Class for natural language profile-based dense retrieval."""

import os

from pyserini.encode import AutoQueryEncoder, TctColBertQueryEncoder
from pyserini.search.faiss import FaissSearcher

from src.models.retriever import RankedList, Retriever

SCIBERT_ENCODER = "allenai/scibert_scivocab_uncased"


class DenseRetriever(Retriever):
    """Dense retrieval with FAISS + dense encoder."""

    def __init__(
        self,
        index_root: str,
        encoder_model: str = SCIBERT_ENCODER,
    ):
        self._index_root = index_root
        if "bert" in encoder_model.lower():
            self._encoder = TctColBertQueryEncoder(encoder_model)
        else:
            self._encoder = AutoQueryEncoder(encoder_model)

    def score(
        self,
        author_id: str,
        nl_profile: str,
        top_k: int = 100,
    ) -> RankedList:
        index_dir = os.path.join(self._index_root, author_id)
        if not os.path.exists(index_dir):
            print(f"[WARN] Index for {author_id} not found, skipping...")
            return []

        searcher = FaissSearcher(index_dir, self._encoder)
        hits = searcher.search(nl_profile, k=top_k)
        return [(hit.docid, float(hit.score)) for hit in hits]
