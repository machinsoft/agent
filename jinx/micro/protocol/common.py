from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class AuthMode(str, Enum):
    apiKey = "apiKey"
    chatgpt = "chatgpt"


@dataclass
class GitSha:
    value: str

    @staticmethod
    def new(sha: str) -> "GitSha":
        return GitSha(sha)


# Fuzzy file search types (minimal parity)
@dataclass
class FuzzyFileSearchParams:
    query: str
    roots: List[str]
    cancellation_token: Optional[str] = None


@dataclass
class FuzzyFileSearchResult:
    root: str
    path: str
    file_name: str
    score: int
    indices: Optional[List[int]] = None


@dataclass
class FuzzyFileSearchResponse:
    files: List[FuzzyFileSearchResult]
