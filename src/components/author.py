"""Representation of data associated with an author."""

import json
from dataclasses import dataclass
from typing import Optional

from src.components.article import Article


@dataclass
class Author:
    """Represents data associated with an author."""

    author_id: str
    author_name: str
    nl_profile_input: list[Article]
    ground_truth_items: list[Article]
    nl_profile: Optional[str] = None
    candidate_items: Optional[list[Article]] = None
    split: int = None

    def __str__(self) -> str:
        """Returns a string representation of the author."""
        return (
            f"AuthorData(id={self.id}, name={self.name}, "
            f"Number of NL profile input items={len(self.nl_profile_input)}, "
            f"Number of ground truth items={len(self.ground_truth_items)})"
        )

    def to_json(
        self,
        attr_list: list[str] = [
            "nl_profile_input",
            "ground_truth_items",
            "nl_profile",
            "candidate_items",
        ],
    ) -> str:
        """Converts the author data to a JSON-formatted string."""
        author_json = {"author_id": self.author_id}
        for attr in attr_list:
            if attr in [
                "nl_profile_input",
                "ground_truth_items",
                "candidate_items",
            ]:
                author_json[attr] = [
                    article.to_dict() for article in getattr(self, attr)
                ]
            else:
                author_json[attr] = getattr(self, attr)

        return json.dumps(author_json)
