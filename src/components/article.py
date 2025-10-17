"""Representations of articles (research papers) and references."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    """Represents an article."""

    article_id: str
    title: str
    abstract: str
    categories: list[str]
    update_date: datetime
    author_names: list[str]
    author_ids: Optional[list[str]] = (
        None  # Only set if author names are resolved to IDs
    )
    abstract: str
    categories: list[str]
    update_date: datetime

    def to_json(self) -> str:
        """Converts the article data to a JSON-formatted string."""
        return json.dumps(
            {
                "article_id": self.article_id,
                "title": self.title,
                "author_names": self.author_names,
                "author_ids": self.author_ids,
                "abstract": self.abstract,
                "categories": self.categories,
                "update_date": self.update_date.isoformat(),
            },
            indent=4,
        )

    def to_dict(self) -> dict:
        """Converts the article object into a dictionary with a JSON-friendly
        format."""
        data = asdict(self)
        data["update_date"] = self.update_date.isoformat()
        return data


@dataclass
class ReferencedArticles:
    """Represents a collection of articles referenced by an article."""

    article_id: str
    referenced_article_ids: list[str]
