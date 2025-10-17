"""Utility functions for file operations, such as reading and writing different
format files."""

import json
from collections import defaultdict

from src.models.retriever import RankedList


def parse_trec_file(file_path: str) -> dict[str, RankedList]:
    """
    Parses results from a TREC-formatted file.

    Args:
        file_path: The path to the TREC results file.

    Returns:
        A dictionary mapping each query ID to a ranked list of
        (document ID, score) tuples.
    """
    trec_data = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            query_id, _, doc_id, _, score, _ = line.strip().split()
            trec_data[query_id].append((doc_id, float(score)))
    return trec_data


def load_document_contents(
    base_path: str, author_id: str, selected_doc_ids: frozenset[str]
) -> dict[str, str]:
    """Loads document texts from jsonl files for a given list of doc_ids.

    Args:
        base_path: Base directory where author folders are located.
        author_id: The author ID whose documents to load.
        selected_doc_ids: A set of document IDs to load.

    Returns:
        A dictionary mapping doc_id to its text content.
    """
    doc_ids_to_contents = {}
    jsonl_path = f"{base_path}/{author_id}/docs.jsonl"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["id"] not in selected_doc_ids:
                continue
            doc_ids_to_contents[doc["id"]] = doc["contents"]

    return doc_ids_to_contents


def load_nl_profiles(dataset_path: str) -> dict[str, str]:
    """Load author_id -> nl_profile (query) from dataset.jsonl."""
    queries = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries[data["author_id"]] = data["nl_profile"]
    return queries


def write_trec_results(
    output_path: str,
    query_id: str,
    ranked_doc_ids: RankedList,
    run_id: str,
):
    """Writes results in TREC format, appending to the file."""
    with open(output_path, "a") as f:
        for rank, (doc_id, score) in enumerate(ranked_doc_ids, start=1):
            f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_id}\n")
