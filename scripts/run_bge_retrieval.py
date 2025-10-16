"""
Per-author document retrieval using FlagReranker.

This script scores all documents for each author based on their natural language 
profile and writes the top-k results in TREC format.

Usage examples:

1. Default usage with CPU and default paths:
   $ python run_bge_retrieval.py

2. Specify custom paths and top-k:
   $ python run_bge_retrieval.py \
       --profiles_path data/dataset.jsonl \
       --docs_base_path data/docs \
       --output_path data/retrieval/results_dense_flag.trec \
       --top_k 100

3. Use GPU (cuda:0) and disable fp16:
   $ python run_bge_retrieval.py \
       --device cuda:0 \
       --no_fp16

4. Use a different FlagReranker model and custom run name:
   $ python run_bge_retrieval.py \
       --model_name BAAI/bge-reranker-base \
       --run_name my_reranker_run
"""

import argparse
import json
import os

from FlagEmbedding import (
    FlagLLMReranker,
    FlagReranker,
    LayerWiseFlagLLMReranker,
    LightWeightFlagLLMReranker,
)
from tqdm import tqdm


def init_model(model_name="BAAI/bge-reranker-large", use_fp16=True, devices=["cpu"]):
    if "v2-gemma" in model_name:
        return FlagLLMReranker(model_name, use_fp16=use_fp16, devices=devices)
    if "v2-minicpm-layerwise" in model_name:
        return LayerWiseFlagLLMReranker(model_name, use_fp16=use_fp16, devices=devices)
    if "v2.5-gemma2-lightweight" in model_name:
        return LightWeightFlagLLMReranker(
            model_name, use_fp16=use_fp16, devices=devices
        )
    return FlagReranker(model_name, use_fp16=use_fp16, devices=devices)


def load_author_profiles(profiles_path):
    profiles = {}
    with open(profiles_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            profiles[entry["author_id"]] = entry["nl_profile"]
    return profiles


def load_author_docs(docs_base_path, author_id):
    """
    Returns a list of unique docs: [{"id": doc_id, "contents": text}, ...]
    """
    docs_path = os.path.join(docs_base_path, "authors", author_id, "docs.jsonl")
    seen = set()
    docs = []
    if os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                doc_id = entry["id"]
                if doc_id not in seen:
                    docs.append(entry)
                    seen.add(doc_id)
    return docs


def score_docs_for_author(model, author_id, nl_profile, docs, top_k=100):
    pairs = [[nl_profile, doc["contents"]] for doc in docs]
    if not pairs:
        return []
    if isinstance(model, LayerWiseFlagLLMReranker):
        scores = model.compute_score(pairs, cutoff_layers=[28])
    elif isinstance(model, LightWeightFlagLLMReranker):
        scores = model.compute_score(
            pairs,
            cutoff_layers=[28],
            compress_ratio=2,
            compress_layers=[24, 40],
        )
    else:
        scores = model.compute_score(pairs)
    scored_docs = [(doc["id"], float(score)) for doc, score in zip(docs, scores)]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]


def write_trec_results(all_results, output_path, run_name="flag_reranker"):
    with open(output_path, "w", encoding="utf-8") as f:
        for author_id, scored_docs in all_results.items():
            for rank, (doc_id, score) in enumerate(scored_docs, start=1):
                f.write(f"{author_id} Q0 {doc_id} {rank} {score} {run_name}\n")


def main(args):
    model = init_model(
        model_name=args.model_name,
        use_fp16=not args.no_fp16,
        devices=args.devices,
    )
    profiles = load_author_profiles(args.profiles_path)

    all_results = {}
    for author_id, nl_profile in tqdm(profiles.items(), desc="Authors"):
        docs = load_author_docs(args.docs_base_path, author_id)
        scored_docs = score_docs_for_author(
            model, author_id, nl_profile, docs, top_k=args.top_k
        )
        all_results[author_id] = scored_docs

    write_trec_results(all_results, args.output_path, run_name=args.run_name)
    print(f"TREC results written to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-author retrieval using FlagReranker"
    )
    parser.add_argument(
        "--profiles_path",
        type=str,
        default="dataset.jsonl",
        help="Path to dataset.jsonl with author profiles",
    )
    parser.add_argument(
        "--docs_base_path",
        type=str,
        default="docs",
        help="Base path to docs folder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results_trec.txt",
        help="Path to write TREC formatted results",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top documents to return per author",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-reranker-large",
        help="FlagReranker model name",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu"],
        help="List of devices to use, e.g. --devices cuda:0 cuda:1",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="flag_reranker",
        help="Run name in TREC file",
    )
    parser.add_argument(
        "--no_fp16", action="store_true", help="Disable fp16 even if available"
    )

    args = parser.parse_args()
    main(args)
