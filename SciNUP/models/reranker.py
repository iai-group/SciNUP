"""Base class for reranking retrieval results."""

from abc import ABC, abstractmethod

from SciNUP.models.retriever import RankedList


class Reranker(ABC):
    """Abstract retriever for NL-profile based recommendation."""

    @abstractmethod
    def rerank(
        self,
        author_id: str,
        query: str,
        retrieval_results: RankedList,
        doc_ids_to_contents: dict[str, str],
    ) -> RankedList:
        """Reranks retrieved items for a given author.

        Args:
            author_id: The author/user identifier.
            query: The user's natural language profile.
            retrieval_results: Initial retrieval results as a list of
                                (docid, score) pairs.
            doc_ids_to_contents: Mapping from document IDs to their contents.

        Returns:
            A list of (docid, score) pairs.
        """
        raise NotImplementedError("The rerank method must be implemented.")
