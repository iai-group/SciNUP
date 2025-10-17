"""Script to rerank retrieval results using Pairwise Relevance Prompting with 
an LLM.

Run 'python rerank_pipeline.py --help' for usage details and default arguments.

All arguments are optional. It's recommended to specify the following three 
arguments given in the example usage below:
    python run_llm_reranking.py \
    --retrieval_results_path input_path \
    --output_file output_path \
    --llm_model_name model_name \
    
IMPORTANT: Ensure your API key is set as an environment variable
In your terminal, run:
export OPENROUTER_API_KEY='your-api-key-here'
"""

import argparse
import os

from src.components import file_utils
from src.models.prp_reranker import PRPReranker

# --- Configuration ---
DATASET_PATH = "data/SciNUP/dataset.jsonl"
DOCS_BASE_PATH = "data/docs/authors"
RETRIEVAL_RESULTS_PATH = "data/retrieval_results/bm25.trec"
OUTPUT_FILE = "data/retrieval_results/reranked_results.trec"
LLM_MODEL_NAME = "openrouter/meta-llama/llama-4-maverick-17b-128e-instruct:free"
SLIDING_K = 10  # Number of sliding window passes for reranking
QUERY_LIMIT = 0  # Set to 0 to process all queries


def main(args):
    """
    Main pipeline to load data, rerank documents, and save the results.
    Accepts command-line arguments for all configurations.
    """
    # Ensure the output file is clear before a new run
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
        print(f"Removed existing output file: {args.output_file}")

    # 1. Load data using paths from args
    print("Loading queries and initial rankings...")
    nl_profiles = file_utils.load_nl_profiles(args.dataset_path)
    initial_rankings = file_utils.parse_trec_file(args.retrieval_results_path)

    # 2. Initialize the reranker model using params from args
    reranker = PRPReranker(llm_model_name=args.llm_model_name, sliding_k=args.sliding_k)

    # 3. Process each query
    queries_processed = 0
    for author_id, initial_rankedlist in initial_rankings.items():
        if author_id not in nl_profiles.keys():
            print(f"Warning: No query found for author_id {author_id}. Skipping.")
            continue

        nl_profile = nl_profiles[author_id]
        print(f"\n--- Reranking for author: {author_id} ---")
        print(f"Query: {nl_profile[:200]}...")

        doc_ids = [doc_id for doc_id, _ in initial_rankedlist]
        doc_ids_to_contents = file_utils.load_document_contents(
            args.docs_base_path, author_id, set(doc_ids)
        )

        reranked_rankedlist = reranker.rerank(
            author_id, nl_profile, initial_rankedlist, doc_ids_to_contents
        )

        file_utils.write_trec_results(
            output_path=args.output_file,
            query_id=author_id,
            ranked_doc_ids=reranked_rankedlist,
            run_id="llm_rerank",
        )
        print(f"Successfully reranked and saved results for {author_id}.")

        queries_processed += 1
        # Check against the query limit from args
        if args.query_limit > 0 and queries_processed >= args.query_limit:
            print(f"\nReached query limit of {args.query_limit}. Halting.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rerank retrieval results using a pairwise relevance " "prompting with LLM."
        ),
        # Shows defaults in help message
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--retrieval_results_path",
        type=str,
        default=RETRIEVAL_RESULTS_PATH,
        help="Path to the initial retrieval results file in TREC format.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=OUTPUT_FILE,
        help="Path to write the final reranked results in TREC format.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DATASET_PATH,
        help="Path to the dataset file containing queries (nl_profile).",
    )
    parser.add_argument(
        "--docs_base_path",
        type=str,
        default=DOCS_BASE_PATH,
        help="Base directory containing the document collections.",
    )

    # --- Model & Reranking Arguments ---
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default=LLM_MODEL_NAME,
        help="The name of the LLM to use for reranking via OpenRouter.",
    )
    parser.add_argument(
        "--sliding_k",
        type=int,
        default=SLIDING_K,
        help="Number of sliding window passes to perform for reranking.",
    )

    # --- Execution Control Arguments ---
    parser.add_argument(
        "--query_limit",
        type=int,
        default=QUERY_LIMIT,
        help="Maximum number of queries to process. Set to 0 to run on all queries.",
    )

    args = parser.parse_args()
    main(args)
