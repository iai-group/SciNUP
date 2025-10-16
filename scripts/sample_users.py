"""Sample users and save their data, including the NL profile input and ground
truth items."""

import ast

import pandas as pd

from SciNUP.components.article import Article, ReferencedArticles
from SciNUP.components.author import Author

# Constants for file paths
CANDIDATE_USERS_CSV_PATH = "data/preprocessed/candidate_users.csv"
METADATA_CSV_PATH = "data/preprocessed/arxiv-metadata.csv"
CITATIONS_CSV_PATH = "data/preprocessed/citations.csv"
SAMPLED_USERS_JSONL_PATH = "data/SciNUP/sampled_users.jsonl"
# Number of sample users to select
N_SAMPLE_USERS = 1050
# Minimum  and maximum number of papers an author must have to be considered
N_MIN_PAPERS = 10
N_MAX_PAPERS = 500
# Random seed for reproducibility
SEED = 42


def sample_users(
    metadata: pd.DataFrame, authors: pd.DataFrame, citations: pd.DataFrame
) -> list[Author]:
    """Takes a random sample of users and transforms it into a list of Author.

    Args:
        metadata: DataFrame containing paper metadata.
        authors: DataFrame containing candidate authors and their papers.
        citations: DataFrame containing citation information.

    Returns:
        List of Author objects.
    """

    users = []
    sample = authors[
        (authors["num_papers"] >= N_MIN_PAPERS)
        & (authors["num_papers"] <= N_MAX_PAPERS)
    ].sample(N_SAMPLE_USERS, random_state=SEED)
    print("Dataset sampled")
    counter = 0
    for i, row in sample.iterrows():
        author_id = row["author_id"]
        author_name = row["author_name"]
        article_ids = row["authored_papers"]
        n_articles = row["num_papers"]

        if len(author_name.split(", ")) >= 10:
            continue

        authored_articles_df = metadata[metadata.article_id.isin(article_ids)]
        authored_articles_df = authored_articles_df.sort_values(
            by="update_date"
        ).reset_index(drop=True)

        user_profile_input = df_to_articles(authored_articles_df[: n_articles // 2])

        seen_article_ids = get_referenced_articles_set(user_profile_input, citations)
        ground_truth_ids = get_referenced_articles_set(
            df_to_articles(authored_articles_df[n_articles // 2 :]), citations
        )
        ground_truth_ids = list(set(ground_truth_ids).difference(set(seen_article_ids)))
        ground_truth = df_to_articles(
            metadata[metadata.article_id.isin(ground_truth_ids)]
        )

        users.append(Author(author_id, author_name, user_profile_input, ground_truth))
        counter += 1
        print(counter)
        print(
            f"""author_id: {author_id}, 
            author_name: {author_name}, 
            num_profile_input: {len(user_profile_input)}, 
            num_ground_truth: {len(ground_truth)}"""
        )
    return users


def df_to_articles(articles_df: pd.DataFrame) -> list[Article]:
    """Converts a DataFrame of articles into a list of Article objects.

    Args:
        articles_df: DataFrame containing metadata about articles.

    Returns:
        List of Article objects.
    """
    articles = []
    for i, article in articles_df.iterrows():
        articles.append(
            Article(
                article_id=article["article_id"],
                title=article["title"],
                author_names=[" ".join(name) for name in article["authors_parsed"]],
                abstract=article["abstract"],
                categories=article["categories"],
                update_date=article["update_date"],
            )
        )
    return articles


def _get_referenced_articles(
    article: Article, citations: pd.DataFrame
) -> ReferencedArticles:
    """Extracts the list of referenced articles for a given article.

    Args:
        article: Article object to find references for.
        citations: DataFrame containing citation information.

    Returns:
        ReferencedArticles object containing unique referenced paper IDs.
    """
    article_id = article.article_id
    if article_id not in citations.article_id.values:
        return ReferencedArticles(article.article_id, [])

    refs = citations[citations.article_id == article_id]["references"].values[0]
    return ReferencedArticles(article_id, list(set(refs)))


def get_referenced_articles_set(
    articles: list[Article], citations: pd.DataFrame
) -> list[str]:
    """Extracts the list of unique referenced article IDs for a list of articles.

    Args:
        articles: List of Article objects to find references for.
        citations: DataFrame containing citation information.

    Returns:
        List of unique referenced article IDs.
    """
    refs = []
    for article in articles:
        refs.extend(_get_referenced_articles(article, citations).referenced_article_ids)
    return list(set(refs))


def main():
    metadata = pd.read_csv(
        METADATA_CSV_PATH,
        dtype={"article_id": "string"},
        converters={"authors_parsed": ast.literal_eval},
        parse_dates=["update_date"],
    )
    print("metadata read")
    authors = pd.read_csv(
        CANDIDATE_USERS_CSV_PATH,
        dtype={"author_name": "string", "num_papers": "int"},
        converters={
            "authored_papers": ast.literal_eval,
            "categories": ast.literal_eval,
        },
    )
    print("authors dataset read")
    citations = pd.read_csv(
        CITATIONS_CSV_PATH,
        dtype={"article_id": "string", "num_references": "int"},
        converters={"references": ast.literal_eval},
    )
    print("citations dataset read")
    users = sample_users(metadata, authors, citations)
    print(f"Number of users: {len(users)}")

    with open(SAMPLED_USERS_JSONL_PATH, "w") as file:
        for author in users:
            file.write(author.to_json() + "\n")

    print(f"Sample users saved to {SAMPLED_USERS_JSONL_PATH}")


if __name__ == "__main__":
    main()
