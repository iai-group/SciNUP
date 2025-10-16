"""Creates dense indices for each author using SciBERT.

Example usage:
    python scripts/build_scibert_index.py \
        --authors_dir authors_dir \
        --output_dir output_dir
"""

import argparse
import os
import subprocess

from tqdm import tqdm


def encode_dense_vectors(
    author_id: str,
    docs_path: str,
    output_dir: str,
    encoder: str,
    device: str = "cpu",
):
    author_output_dir = os.path.join(output_dir, author_id)
    os.makedirs(author_output_dir, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "pyserini.encode",
        "input",
        "--corpus",
        docs_path,
        "--fields",
        "text",
        "--delimiter",
        "\n\n",
        "output",
        "--embeddings",
        author_output_dir,
        "--to-faiss",
        "encoder",
        "--encoder",
        encoder,
        "--fields",
        "text",
        "--device",
        device,
    ]
    subprocess.run(cmd, check=True)


def main(
    authors_dir: str,
    output_dir: str,
    encoder: str,
    device: str,
):

    author_ids = [
        d
        for d in os.listdir(authors_dir)
        if os.path.isdir(os.path.join(authors_dir, d))
    ]

    for author_id in tqdm(author_ids, desc="Encoding dense vectors"):
        docs_path = os.path.join(authors_dir, author_id)
        encode_dense_vectors(author_id, docs_path, output_dir, encoder, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode dense vectors for author documents."
    )
    parser.add_argument(
        "--authors_dir",
        required=True,
        help="Path to directory containing author folders with documents",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to store encoded vectors",
    )
    parser.add_argument(
        "--encoder",
        default="allenai/scibert_scivocab_uncased",
        help="Encoder model to use",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run encoding on",
    )

    args = parser.parse_args()
    main(args.authors_dir, args.output_dir, args.encoder, args.device)
