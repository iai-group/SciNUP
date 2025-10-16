"""Preprocesses the data."""

import ast
import json
import re
import unicodedata

import pandas as pd

# Constants for file paths
CITATIONS_JSON_PATH = "data/raw/internal-references-v0.2.0-2019-03-01.json"
CITATIONS_CSV_PATH = "data/preprocessed/citations.csv"
KAGGLE_METADATA_PATH = "data/raw/arxiv-metadata-oai-snapshot.json"
GITHUB_METADATA_PATH = "data/raw/arxiv-metadata-hash-abstracts-v0.2.0-2019-03-01.json"
METADATA_CSV_PATH = "data/preprocessed/arxiv-metadata.csv"
AUTHORS_CSV_PATH = "data/preprocessed/authors.csv"
CANDIDATE_USERS_CSV_PATH = "data/preprocessed/candidate_users.csv"
N_MINIMUM_PAPERS = 5


def main() -> None:
    # Citations
    preprocess_citations(CITATIONS_JSON_PATH, CITATIONS_CSV_PATH)
    citations = pd.read_csv(
        CITATIONS_CSV_PATH,
        dtype={"article_id": "string", "num_references": "int"},
    )
    citations["references"] = citations["references"].apply(ast.literal_eval)

    # Metadata
    preprocess_metadata(METADATA_CSV_PATH, citations.article_id.tolist())

    # Authors
    _create_authors_df(METADATA_CSV_PATH, AUTHORS_CSV_PATH)
    _filter_authors(CANDIDATE_USERS_CSV_PATH)


def preprocess_metadata(output_path: str, citation_ids: list[str]) -> None:
    """Merges github and Kaggle versions of arxiv datasets and saves the merged
    dataframe to a CSV file.

    Args:
        output_path: Path to save the merged CSV file.
        citation_ids: List of article IDs present in citations dataframe.
    """

    # Read ArXiv metadata github & Kaggle versions
    print("preprocessing metadata...")
    df = pd.read_json(KAGGLE_METADATA_PATH, lines=True)
    df["id"] = df["id"].apply(lambda x: str(x).strip())
    df1 = pd.read_json(GITHUB_METADATA_PATH, lines=True)
    df1["id"] = df1["id"].apply(lambda x: str(x).strip())

    # Merge them and make sure there are no duplicates
    df = df.merge(df1[["id"]], on="id")
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)

    # Fill in date column and format it correctly
    df["update_date"] = pd.to_datetime(df["update_date"])
    df.update_date = df.update_date.where(
        df.update_date.notnull(),
        df.versions.apply(lambda x: pd.to_datetime(x[0]["created"])),
    )

    # Filter the dataframe to include only the IDs present in the citations
    # dataframe and save it to the same path
    df = df[df.id.isin(citation_ids)].reset_index(drop=True)
    df.rename(columns={"id": "article_id"}, inplace=True)
    df.to_csv(output_path, index=False)
    print("metadata preprocessed and saved to ", output_path)


def preprocess_citations(json_path: str, output_path: str) -> None:
    """Preprocesses citations data by loading the JSON file, extracts relevant
    information and saves it to a CSV file.

    Args:
        json_path: Path to the input JSON file containing citations data.
        output_path: Path to save the preprocessed CSV file.
    """
    print("preprocessing citations...")
    with open(json_path) as f:
        refs = json.load(f)

    citations = pd.DataFrame(refs.items(), columns=["id", "references"])
    citations["num_references"] = citations["references"].apply(len)
    citations = citations[citations["num_references"] > 0].reset_index(drop=True)
    citations["id"] = citations["id"].apply(lambda x: str(x).strip())
    citations.rename(columns={"id": "article_id"}, inplace=True)
    citations.to_csv(output_path, index=False)
    print("citations data preprocessed and saved to", output_path)


def _create_authors_df(metadata_csv_path: str, output_path: str) -> None:
    """Creates a DataFrame of authors from the metadata CSV file, grouping their
    papers and categories as lists into corresponding columns.

    Args:
        output_path: Path to save the authors DataFrame.
    """
    print("creating authors dataframe...")
    df = pd.read_csv(metadata_csv_path, dtype={"article_id": "string"})
    authors = df[["article_id", "authors_parsed", "categories"]]
    authors.rename(columns={"article_id": "authored_paper_ids"}, inplace=True)
    authors.loc[:, "authors_parsed"] = authors["authors_parsed"].apply(ast.literal_eval)
    authors.loc[:, "categories"] = authors["categories"].apply(
        lambda x: str(x).strip().split(" ")
    )
    author = (
        authors.apply(lambda x: pd.Series(x["authors_parsed"]), axis=1)
        .stack()
        .reset_index(level=1, drop=True)
    )
    author.name = "author_name"
    authors = authors.drop("authors_parsed", axis=1).join(author)
    authors["author_name"] = pd.Series(authors["author_name"], dtype=object)
    authors["author_name"] = authors["author_name"].apply(lambda x: " ".join(x))

    # Normalize author names
    authors["author_name"] = authors["author_name"].apply(_normalize_author_name)
    authors = authors[authors["author_name"].apply(_is_valid_name)]

    # Create id column and group by it
    authors["author_id"] = authors["author_name"].apply(
        lambda x: _generate_base_id(x) + "_1"
    )
    authors = authors.groupby("author_id").agg(list).reset_index()
    authors.loc[:, "categories"] = authors["categories"].apply(
        lambda x: set([i for y in x for i in y])
    )
    authors.loc[:, "author_name"] = authors["author_name"].apply(set)

    authors["num_papers"] = authors["authored_paper_ids"].apply(len)
    authors.reset_index(drop=True, inplace=True)

    authors.to_csv(AUTHORS_CSV_PATH, index=False)
    print("authors data preprocessed and saved to", output_path)


def _filter_authors(output_path: str, n_minimum_papers: int = N_MINIMUM_PAPERS) -> None:
    """Filters authors with a minimum number of papers and saves the filtered
    DataFrame.

    Args:
        output_path: Path to save the filtered authors DataFrame.
        n_minimum_papers: Minimum number of papers for an author to be included.
    """
    print("filtering authors...")
    authors = pd.read_csv(AUTHORS_CSV_PATH)

    authors = authors[authors.num_papers >= n_minimum_papers]

    authors.to_csv(output_path, index=False)
    print("authors data filtered and saved to", output_path)


def _is_valid_name(name: str) -> bool:
    """Filters valid names.

    Arg
        name: Author name to be checked.

    Returns:
        True if the name is considered valid, False otherwise.
    """
    if not name or not isinstance(name, str):
        return False

    # Check for presence of alphabetic characters
    if not any(char.isalpha() for char in name):
        return False

    # Count alphabetic tokens (e.g., "J. Smith" → ['J', 'Smith'])
    tokens = [t for t in re.split(r"\W+", name) if any(c.isalpha() for c in t)]
    if len(tokens) < 2:
        return False

    # Check for presence of alphabetic characters
    if any(char.isdigit() for char in name):
        return False

    # Check for too many comma-separated segments (likely an affiliation line)
    if name.count(",") > 5 or len(name.split()) > 5:
        return False

    return True


def _normalize_author_name(name: str) -> str:
    """Normalizes an author name: lowercasing, removing punctuation, collapsing
    whitespace, converting accents.

    Args:
        name: Author name to be normalized.

    Returns:
        Normalized author name as a string.
    """
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = name.replace("collaboration", "")
    name = re.sub(r"[^\w\s]", "", name)  # remove punctuation
    name = re.sub(r"\s+", " ", name).strip()  # remove excess whitespace
    return name


def _generate_base_id(name: str) -> str:
    tokens = name.split()
    if len(tokens) >= 2:
        surname = tokens[0]
        first_initial = tokens[1][0]
        return f"{surname}_{first_initial}"
    return "unknown"


if __name__ == "__main__":
    main()
