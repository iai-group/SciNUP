"""Run retrieval based on NL profiles using either sparse or dense methods.

Example usage:
    python src/components/run_retrieval.py \
    --dataset data/dataset.jsonl \
    --output data/retrieval/results_bm25.txt \
    --method sparse \
    --index_root data/indexes/sparse/authors \
    --sparse_method bm25 \
    --top_k 100 \
    --run_name bm25_run
"""

import argparse
import json

from tqdm import tqdm

from SciNUP.models.dense_retriever import DenseRetriever
from SciNUP.models.sparse_retriever import SparseRetriever


def load_profiles(dataset_path: str) -> list[dict]:
    """Loads dataset including NL profiles from a JSONL file.

    Args:
        dataset_path: Path to dataset JSONL file containing NL profiles.

    Returns:
        List of dictionaries, each containing author_id, nl_profile and
        candidate_items.
    """
    with open(dataset_path, "r") as f:
        return [json.loads(line) for line in f]


def write_results_to_trec_file(
    output_file: str, run_name: str, author_id: str, results: tuple[str, float]
) -> None:
    """Writes retrieval results to a TREC-formatted file.

    Args:
        output_file: Path to the output file.
        run_name: Name of the run, used for TREC format.
        author_id: Identifier for the author/user.
        results: List of tuples containing (docid, score) pairs.
    """
    seen = set()
    rank = 1
    with open(output_file, "a") as f:  # append mode
        for docid, score in sorted(results, key=lambda x: x[1], reverse=True):
            if docid in seen:
                continue
            seen.add(docid)
            f.write(f"{author_id} Q0 {docid} {rank} {score:.4f} {run_name}\n")
            rank += 1


def main():
    parser = argparse.ArgumentParser(description="Run retrieval experiments.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset JSONL file with NL profiles.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save results."
    )
    parser.add_argument(
        "--method",
        choices=["sparse", "dense"],
        required=True,
        help="Retrieval method.",
    )
    parser.add_argument(
        "--index_root",
        type=str,
        required=True,
        help="Root directory of indexes.",
    )
    parser.add_argument(
        "--top_k", type=int, default=100, help="Number of results per author."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="retrieval_run",
        help="Name for the run.",
    )
    parser.add_argument(
        "--sparse_method",
        type=str,
        default="bm25",
        help="Sparse method: bm25, qld, rm3.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="Dense encoder model.",
    )
    args = parser.parse_args()

    if args.method == "sparse":
        retriever = SparseRetriever(
            index_root=args.index_root, method=args.sparse_method
        )
    elif args.method == "dense":
        retriever = DenseRetriever(
            index_root=args.index_root, encoder_model=args.encoder
        )
    else:
        raise ValueError(f"Unknown method {args.method}")

    dataset = load_profiles(args.dataset)

    # clear output file before writing
    open(args.output, "w").close()

    for user in tqdm(dataset, desc="Running retrieval"):
        author_id = user["author_id"]
        nl_profile = user["nl_profile"]
        results = retriever.score(author_id, nl_profile, top_k=args.top_k)
        if results:
            write_results_to_trec_file(args.output, args.run_name, author_id, results)


if __name__ == "__main__":
    main()
