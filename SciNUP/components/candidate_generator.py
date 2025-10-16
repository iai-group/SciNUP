"""CandidateGenerator class for generating candidate items for a user."""

import numpy as np
import pandas as pd

from SciNUP.components.article import Article
from SciNUP.components.author import Author
from scripts.sample_users import df_to_articles, get_referenced_articles_set

# Number of candidate items to sample
NUM_CANDIDATES = 1000
# Random state for reproducibility
SEED = 40


class CandidateGenerator:
    def __init__(
        self,
        metadata: pd.DataFrame,
        citations: pd.DataFrame,
    ):
        """Initializes the CandidateGenerator.

        Args:
            metadata: DataFrame containing paper metadata.
            citations: DataFrame containing citation information.
        """
        self._metadata = metadata
        self._citations = citations

    def generate_candidates(self, author: Author) -> list[Article]:
        """Generates candidate items for the user.

        N-plus-random candidates are generated, where N is the number of ground
        truth items and random items are sampled from the articles in the same
        categories that the user has been writing and citing from before.

        Args:
            author: Author object corresponding the user.

        Returns:
            List of candidate items for the user.
        """
        profile_input_articles = author.nl_profile_input
        seen_article_ids = get_referenced_articles_set(
            profile_input_articles, self._citations
        )

        ground_truth_articles = author.ground_truth_items
        print("num_ground_truth_articles: ", len(ground_truth_articles))
        authored_categories = np.concatenate(
            [article.categories for article in profile_input_articles],
            axis=None,
        )
        categories = {
            item: authored_categories.tolist().count(item) / len(authored_categories)
            for item in set(authored_categories)
        }

        candidate_items = self._n_plus_random(
            categories=categories,
            seen_article_ids=seen_article_ids,
            ground_truth_articles=ground_truth_articles,
            num_candidates=NUM_CANDIDATES,
        )
        return candidate_items

    def _n_plus_random(
        self,
        categories: dict[str, float],
        seen_article_ids: list[str] = [],
        ground_truth_articles: list[Article] = [],
        num_candidates: int = NUM_CANDIDATES,
    ) -> list[Article]:
        """Generates a set of candidate items using the N-plus-random strategy.

        N is the number of ground truth items. Remaining items are random items
        sampled from the union of provided categories.

        Args:
            categories: Dictionary of categories with their weights.
            num_candidates: Number of candidates items.
            seen_article_ids: List of article IDs that have been seen by the user.
            ground_truth_articles: List of ground truth articles for the user.

        Returns:
            List of candidate items.
        """
        candidate_items = []
        candidate_items.extend(ground_truth_articles)
        filtered_df = self._metadata[~self._metadata.article_id.isin(seen_article_ids)]
        num_to_sample = num_candidates - len(candidate_items)
        if num_to_sample < 0:
            print(
                f"WARNING: Ground truth articles are more than {num_candidates}:"
                f" {len(candidate_items)}"
            )

        for category, weight in categories.items():
            cat_df = filtered_df[
                filtered_df.categories.astype(str).str.contains(category)
            ]
            n = int(num_to_sample * weight)
            if n == 0:
                continue

            sample = cat_df.sample(n, random_state=SEED) if len(cat_df) >= n else cat_df
            candidate_items.extend(df_to_articles(sample))

            filtered_df = filtered_df[~filtered_df.article_id.isin(sample.article_id)]

        if len(candidate_items) < num_candidates:
            print(
                "Sampling from all the categories to get enough number of " "candidates"
            )
            cat_df = filtered_df[
                filtered_df.categories.astype(str).str.contains("|".join(categories))
            ]
            n = num_candidates - len(candidate_items)
            sample = cat_df.sample(n, random_state=SEED) if len(cat_df) >= n else cat_df
            candidate_items.extend(df_to_articles(sample))

        if len(candidate_items) < num_candidates:
            print("Sampling from the dataset without category filtering")
            sample = filtered_df.sample(num_candidates - len(candidate_items))
            candidate_items.extend(df_to_articles(sample))

        return sorted(candidate_items, key=lambda article: article.article_id)
