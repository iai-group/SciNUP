"""Script that prompts an LLM to classify natural language profiles into narrow,
medium, or broad categories.

Sample usage:
    python classify_nl_profiles.py \
    --dataset_path data/dataset.jsonl \
    --output_file data/nl_profiles_classifications.jsonl \
    --llm_model_name openrouter/meta-llama/llama-3-8b-instruct

If backend is ollama:
    python classify_nl_profiles.py \
    --dataset_path data/dataset.jsonl \
    --output_file data/nl_profiles_classifications.jsonl \
    --llm_model_name llama3.3:70b \
    --backend ollama
"""

import argparse
import json
import os

from src.components import file_utils, llm_utils

_PROMPT = (
    "You are an expert in classifying scholarly user interest profiles. "
    "Your task is to analyze a given natural language description of a user's "
    "research interests and classify it as 'narrow', 'medium', or 'broad' based "
    "on the specificity and scope of the topics mentioned.\n\n"
    "Here are the definitions for each category:\n\n"
    "Narrow: The profile describes highly specific interests within a single, "
    "well-defined subfield. The language is often technical and domain-specific."
    "\nMedium: The profile covers a single, broader field or several related "
    "topics. The interests are connected but not as specific as a narrow profile."
    "\nBroad: The profile covers a wide range of disparate topics or a very "
    "general field. The interests may not be directly connected.\n\n"
    "Examples:\n\n"
    "User Profile: 'My research focuses on the optimization of federated learning "
    "algorithms for on-device natural language processing, specifically for "
    "low-resource languages.'\n"
    "Classification: Narrow\n\n"
    "User Profile: 'I am interested in the intersection of artificial "
    "intelligence and medicine. My work involves using computer vision for "
    "medical image analysis and developing predictive models for disease "
    "progression using electronic health records.'\n"
    "Classification: Medium\n\n"
    "User Profile: 'I have a passion for technology and its role in society. I'm"
    " interested in everything from robotics and human-computer interaction to "
    "the ethical implications of AI and the future of work. I also enjoy "
    "historical perspectives on technological innovation.'\n"
    "Classification: Broad\n\n"
    "Please classify the following user profile. Return only one word, Narrow, "
    "Medium or Broad\n\n"
    "User Profile: {nl_profile}\n"
    "Classification:"
)

DATASET_PATH = "data/SciNUP/dataset.jsonl"
OUTPUT_FILE = "data/nl_profiles_classifications"
LLM_MODEL_NAME = "openrouter/meta-llama/llama-3.3-8b-instruct:free"


def main(args):
    """
    Main pipeline to load data, rerank documents, and save the results.
    Accepts command-line arguments for all configurations.
    """

    nl_profiles = file_utils.load_nl_profiles(args.dataset_path)

    # clear output files if they exist
    if os.path.exists(args.output_file + ".jsonl"):
        os.remove(args.output_file + ".jsonl")
        print(f"Removed existing output file: {args.output_file}.jsonl")
    if os.path.exists(args.output_file + ".tsv"):
        os.remove(args.output_file + ".tsv")
        print(f"Removed existing output file: {args.output_file}.tsv")

    for author_id, nl_profile in nl_profiles.items():
        print(f"\n--- Classifying profile for author: {author_id} ---")
        print(f"Profile: {nl_profile[:200]}...")

        prompt = _PROMPT.format(nl_profile=nl_profile)

        classification = llm_utils.call_llm(
            prompt=prompt,
            model=args.llm_model_name,
            backend=args.backend,
            max_tokens=5,
        )

        print(f"Classification: {classification}")

        # Save classification result
        with open(args.output_file + ".jsonl", "a") as f:
            json.dump(
                {
                    "author_id": author_id,
                    "nl_profile": nl_profile,
                    "classification": classification,
                },
                f,
            )
            f.write("\n")
        print(f"Saved classification for {author_id} to {args.output_file}.jsonl")
        # Save as tsv for easier viewing
        with open(args.output_file + ".tsv", "a") as f:
            f.write(f"{author_id}\t{classification}\n")
        print(f"Saved classification for {author_id} to {args.output_file}.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rerank retrieval results using a pairwise relevance " "prompting with LLM."
        ),
        # Shows defaults in help message
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--llm_model_name",
        type=str,
        default=LLM_MODEL_NAME,
        help="The name of the LLM to use for reranking via OpenRouter.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openrouter", "ollama"],
        default="openrouter",
        help="Backend to use for LLM inference: 'openrouter' or 'ollama'.",
    )

    args = parser.parse_args()
    main(args)
