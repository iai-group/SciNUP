"""Class for PRP-based reranking of retrieved documents.

Reference:
Qin et al. Large Language Models are Effective Text Rankers with Pairwise Ranking
Prompting, NAACL Findings 2024 (https://aclanthology.org/2024.findings-naacl.97/)
"""

from src.components import llm_utils
from src.models.reranker import Reranker
from src.models.retriever import RankedList

_PROMPT = (
    "You are an expert at judging document relevance. For the given "
    "query, choose which of the two documents is more relevant. "
    "Respond ONLY with A or B.\n\n"
    "Query: {query}\n\n"
    "Document A: {docA}\n\n"
    "Document B: {docB}\n\n"
    "Answer (A or B):"
)


class PRPReranker(Reranker):
    """
    Reranks a list of documents for a given query using a pairwise
    relevance prompting approach with an LLM.
    """

    def __init__(self, llm_model_name: str, sliding_k: int = 10, backend="openrouter"):
        """
        Initializes the reranker.

        Args:
            llm_model_name: The identifier for the LLM to be used for comparisons.
            sliding_k: The number of sliding window passes to perform.
        """
        self._llm_model_name = llm_model_name
        self._sliding_k = sliding_k
        self._backend = backend

    def _make_pairwise_prompt(self, query: str, docA: str, docB: str) -> str:
        """Generate a prompt for pairwise comparison."""
        return _PROMPT.format(query=query, docA=docA, docB=docB)

    def _compare_pair(self, query: str, docA: str, docB: str) -> str:
        """
        Uses the LLM to compare two documents for relevance to the query.
        Caches results to avoid redundant API calls.
        """

        prompt = self._make_pairwise_prompt(query, docA, docB)

        choice = llm_utils.call_llm(
            prompt=prompt, model=self._llm_model_name, backend=self._backend
        )

        return choice

    def rerank(
        self,
        author_id: str,
        query: str,
        retrieval_results: RankedList,
        doc_ids_to_contents: dict[str, str],
    ) -> RankedList:
        """
        Reranks a list of document IDs based on their text content.

        Args:
            author_id: The author ID whose documents are being reranked.
            query: The user query string.
            retrieval_results: RankedList, containing (document ID, score) tuples.
            doc_ids_to_contents: Mapping from document IDs to their contents.

        Returns:
            Updated RankedList with reordered (document ID, score) tuples.
        """
        n = len(retrieval_results)
        current_rankedlist = retrieval_results.copy()

        for pass_num in range(self._sliding_k):
            print(f"\n--- Pass {pass_num + 1}/{self._sliding_k} ---")
            swapped = False
            # Bottom-up comparison (least to most relevant)
            for i in range(n - 2, -1, -1):
                docA_id = current_rankedlist[i][0]
                docB_id = current_rankedlist[i + 1][0]

                winner = self._compare_pair(
                    query,
                    doc_ids_to_contents.get(docA_id),
                    doc_ids_to_contents.get(docB_id),
                )

                if winner == "B":
                    current_rankedlist[i], current_rankedlist[i + 1] = (
                        current_rankedlist[i + 1],
                        current_rankedlist[i],
                    )
                    swapped = True

            if not swapped:
                # Early exit: No swaps in this pass, ranking is stable.
                break

        return [
            (doc_id, len(current_rankedlist) + 1 - idx)
            for idx, (doc_id, _) in enumerate(current_rankedlist)
        ]
