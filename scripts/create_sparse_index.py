"""Creates sparse indexes for each author's candidate items in a given
directory.

Example usage:
    python scripts/create_sparse_index.py \
        --authors_dir authors_dir \
        --output_dir output_dir
"""

import argparse
import os
import subprocess

from tqdm import tqdm


def create_index(author_id: str, docs_path: str, output_dir: str) -> None:
    """Creates a sparse index for a given author's documents.

    Args:
        author_id: Unique identifier for the author.
        docs_path: Path to the directory containing the author's documents.
        output_dir: Root directory where the index will be stored.
    """
    index_dir = os.path.join(output_dir, author_id)

    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    cmd = [
        "python",
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        docs_path,
        "--index",
        index_dir,
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "2",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    subprocess.run(cmd)


def main(authors_dir: str, output_dir: str):
    author_ids = [
        d
        for d in os.listdir(authors_dir)
        if os.path.isdir(os.path.join(authors_dir, d))
    ]

    for author_id in tqdm(author_ids, desc="Indexing authors"):
        docs_path = os.path.join(authors_dir, author_id)
        create_index(author_id, docs_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sparse indexes for authors.")
    parser.add_argument(
        "--authors_dir",
        required=True,
        help="Path to authors' documents directory",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Root directory to store indexes"
    )

    args = parser.parse_args()
    main(args.authors_dir, args.output_dir)
