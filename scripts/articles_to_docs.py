"""Script to convert articles metadata file to author-level docs.json files
for indexing."""

import json
import logging
import os

DATASET_JSONL_PATH = "data/SciNUP/dataset.jsonl"
DOCS_JSONL_PATH = "data/docs/authors/{}/docs.jsonl"


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the articles_to_docs script.")

    with open(DATASET_JSONL_PATH) as f:
        data = [json.loads(line) for line in f]
    logging.info(f"Read {len(data)} authors data from the dataset JSONL file.")

    for record in data:
        author_docs_path = DOCS_JSONL_PATH.format(record["author_id"])
        os.makedirs(os.path.dirname(author_docs_path), exist_ok=True)
        with open(author_docs_path, "w") as file:
            for article in record["candidate_items"]:
                article_id = article["article_id"]
                title = article["title"].strip()
                abstract = article["abstract"].strip()
                contents = f"Title: {title} Abstract: {abstract} \n\n"
                json_line = {"id": article["article_id"], "contents": contents}
                file.write(json.dumps(json_line) + "\n")
                logging.debug(f"{article_id} written to docs JSONL file")
            logging.info(f"File written to {author_docs_path}")


if __name__ == "__main__":
    main()
