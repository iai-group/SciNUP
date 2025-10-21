"""Generates a dataset for NL profile-based recommendations from a set of sampled
users."""

import argparse
import ast
import json
import logging
import time
from datetime import datetime

import pandas as pd
from openai import RateLimitError
from tqdm import tqdm

from src.components.article import Article
from src.components.author import Author
from src.components.candidate_generator import CandidateGenerator
from src.components.profile_generator import ProfileGenerator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CANDIDATE_USERS_CSV_PATH = "data/preprocessed/candidate_users.csv"
METADATA_CSV_PATH = "data/preprocessed/arxiv-metadata.csv"
CITATIONS_CSV_PATH = "data/preprocessed/citations.csv"
INPUT_JSONL_PATH = "data/SciNUP/sampled_users.jsonl"
AUTHORS_SPLIT_CSV_PATH = "data/SciNUP/authors_split.csv"
DATASET_JSONL_PATH = "data/SciNUP/dataset.jsonl"

PROFILE_GENERATION_PROMPT_A = (
    "Below are the titles and abstracts of scientific papers I have authored. "
    "Based on this information, generate a concise first-person research "
    "profile that summarizes my main research interests and areas of expertise. "
    "The profile should be written in no more than three sentences and should "
    "clearly identify the key themes, research trends, and topics I focus on. "
    "The purpose of this profile is to support a scientific literature "
    "recommendation system, so it should accurately reflect my research focus "
    "to help match me with relevant publications. \n\n"
    "This is the list of my publications:"
    "\n\n{}"
    "Write the profile in first person. "
    "Return nothing but the generated profile."
)

PROFILE_GENERATION_PROMPT_B = (
    "Below are the titles and abstracts of scientific papers I have authored. "
    "Based on this information, generate a concise description of my research "
    "interests, characterizing the key topics and areas of expertise."
    "This is the list of my publications:"
    "\n\n{}"
    "Write the profile in first person. "
    "Return nothing but the generated profile."
)

SPLIT_DICT = {
    0: {
        "model": "openrouter/meta-llama/llama-4-maverick-17b-128e-instruct:free",
        "prompt": PROFILE_GENERATION_PROMPT_A,
    },
    1: {
        "model": "openrouter/meta-llama/llama-4-maverick-17b-128e-instruct:free",
        "prompt": PROFILE_GENERATION_PROMPT_B,
    },
    2: {
        "model": "openrouter/openai/chatgpt-4o-latest",
        "prompt": PROFILE_GENERATION_PROMPT_A,
    },
    3: {
        "model": "openrouter/openai/chatgpt-4o-latest",
        "prompt": PROFILE_GENERATION_PROMPT_B,
    },
}


def generate_dataset(
    authors_list: list[Author],
    metadata: pd.DataFrame,
    citations_df: pd.DataFrame,
    output_json_file: str = DATASET_JSONL_PATH,
) -> None:
    """Reads sampled users and generates dataset including NL profiles and
    candidate items for each user.

    Args:
        authors: list of sampled Authors
        metadata: arxiv metadata dataframe
        authors_df: authors dataframe
        citations_df: citations dataframe
        output_json_file: path to save the generated dataset
    """

    candidate_generator = CandidateGenerator(metadata=metadata, citations=citations_df)
    profile_generator = ProfileGenerator()
    authors_split = pd.read_csv(AUTHORS_SPLIT_CSV_PATH, dtype={"author_id": "string"})
    author_split_dict = dict(zip(authors_split["author_id"], authors_split["split"]))

    logger.info(f"Generating dataset from {len(authors_list)} sampled users.")
    with open(output_json_file, "w") as file:
        for author in tqdm(authors_list, desc="Generating dataset"):
            author.split = author_split_dict.get(author.author_id)
            logger.info(f"Processing user {author.author_id} with split {author.split}")
            model_name = SPLIT_DICT[author.split]["model"]
            profile_prompt = SPLIT_DICT[author.split]["prompt"]

            candidate_items = candidate_generator.generate_candidates(author)
            logger.info(f"Candidates generated for user {author.author_id}")
            author.candidate_items = candidate_items
            try:
                profile = profile_generator.generate_profile(
                    author.nl_profile_input, profile_prompt, model_name
                )
                logger.info(
                    f"NL profile generated for user {author.author_id}: "
                    + f"using model {model_name}"
                )
            except RateLimitError as e:
                logger.info("API request limit reached.")
                print(e)
                profile = "Profile was not able to generate"
            logger.info(profile)
            author.nl_profile = profile
            time.sleep(3.1)
            file.write(
                author.to_json(["nl_profile", "candidate_items", "split"]) + "\n"
            )

    logger.info("Dataset generation completed")
    logger.info(f"saved to {output_json_file}")


def read_sampled_users(
    input_jsonl_path: str = INPUT_JSONL_PATH,
) -> list[Author]:

    with open(input_jsonl_path) as f:
        data = [json.loads(line, object_hook=_date_hook) for line in f]

    logger.info(f"Read {len(data)} sampled users' data.")
    authors = [
        Author(
            author["author_id"],
            author["author_name"],
            [Article(**article) for article in author["nl_profile_input"]],
            [Article(**article) for article in author["ground_truth_items"]],
        )
        for author in data
    ]
    return authors


def _date_hook(obj):
    if "update_date" in obj and isinstance(obj["update_date"], str):
        try:
            obj["update_date"] = datetime.fromisoformat(obj["update_date"])
        except ValueError:
            logger.debug("update_date is not ISO format")
            pass
    return obj


def parse_args() -> argparse.Namespace:
    """Parses arguments from command-line call.

    Returns:
        Arguments from command-line call.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        dest="debug",
        help="Debugging mode",
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    metadata = pd.read_csv(
        METADATA_CSV_PATH,
        dtype={"article_id": "string"},
        converters={"authors_parsed": ast.literal_eval},
        parse_dates=["update_date"],
    )
    logger.info("metadata read")

    citations_df = pd.read_csv(
        CITATIONS_CSV_PATH,
        dtype={"article_id": "string", "num_references": "int"},
        converters={"references": ast.literal_eval},
    )
    logger.info("citations dataset read")

    authors_list = read_sampled_users()

    generate_dataset(authors_list, metadata, citations_df)
