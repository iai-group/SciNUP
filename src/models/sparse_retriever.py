"""Class for natural language profile-based sparse retrieval."""

import os

from pyserini.search.lucene import LuceneSearcher

from src.models.retriever import RankedList, Retriever


class SparseRetriever(Retriever):
    """BM25/QL/RM3 sparse retriever."""

    def __init__(self, index_root: str, method: str = "bm25"):
        self.index_root = index_root
        self.method = method.lower()

    def score(
        self,
        author_id: str,
        nl_profile: str,
        top_k: int = 100,
    ) -> RankedList:
        index_dir = os.path.join(self.index_root, author_id)
        if not os.path.exists(index_dir):
            print(f"[WARN] Index for {author_id} not found, skipping...")
            return []

        searcher = LuceneSearcher(index_dir)

        if self.method == "bm25":
            searcher.set_bm25()
        elif self.method == "qld":
            searcher.set_qld()
        elif self.method == "rm3":
            searcher.set_rm3()
        else:
            raise ValueError(f"Unknown sparse method: {self.method}")

        hits = searcher.search(nl_profile, k=top_k)
        return [(hit.docid, hit.score) for hit in hits]
