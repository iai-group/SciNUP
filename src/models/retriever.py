"""Base class for natural language profile-based retrieval."""

from abc import ABC, abstractmethod

RankedList = list[tuple[str, float]]


class Retriever(ABC):
    """Abstract retriever for NL-profile based recommendation."""

    @abstractmethod
    def score(
        self,
        author_id: str,
        nl_profile: str,
        top_k: int = 100,
    ) -> RankedList:
        """Score candidate items for a given author.

        Args:
            author_id: The author/user identifier.
            nl_profile: The user's natural language profile.
            top_k: Number of top results to return.

        Returns:
            A list of (docid, score) pairs.
        """
        raise NotImplementedError("The score method must be implemented.")
