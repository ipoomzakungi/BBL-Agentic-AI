"""Deterministic local retrieval for the knowledge-base text file."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
CHUNK_ID_PATTERN = re.compile(r"^\[([A-Z0-9-]+)\]$")

# Domain-generic words do not help distinguish one policy chunk from another.
STOP_WORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "any",
    "are",
    "be",
    "before",
    "bring",
    "can",
    "do",
    "does",
    "employee",
    "employees",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "policy",
    "should",
    "the",
    "their",
    "to",
    "what",
    "when",
    "who",
    "with",
}

# Small, inspectable query expansion keeps the challenge independent of an
# embedding endpoint while handling common user/document vocabulary differences.
QUERY_EXPANSIONS = {
    "abroad": {"foreign", "international", "travel"},
    "claim": {"expense", "reimbursement", "submit"},
    "claims": {"expense", "reimbursement", "submit"},
    "cost": {"expense", "reimbursement"},
    "costs": {"expense", "reimbursement"},
    "flight": {"booking"},
    "flights": {"flight", "booking"},
    "hotel": {"accommodation"},
    "hotels": {"accommodation"},
    "overseas": {"foreign", "international", "travel"},
    "paperwork": {"documents", "documentation"},
    "receipt": {"expense", "reimbursement"},
    "receipts": {"receipt", "expense", "reimbursement"},
    "trip": {"travel"},
    "trips": {"travel"},
}

TOKEN_NORMALIZATIONS = {
    "approvals": "approval",
    "claims": "claim",
    "documents": "document",
    "documentation": "document",
    "expenses": "expense",
    "flights": "flight",
    "hotels": "hotel",
    "receipts": "receipt",
    "returned": "return",
    "returning": "return",
    "returns": "return",
    "submission": "submit",
    "submitted": "submit",
    "trips": "trip",
}

GENERIC_QUERY_TERMS = {"business", "travel", "trip"}


@dataclass(frozen=True)
class KnowledgeChunk:
    """One searchable knowledge-base record."""

    chunk_id: str
    title: str
    tags: tuple[str, ...]
    content: str

    @property
    def searchable_text(self) -> str:
        """Weight titles and tags slightly without changing returned evidence."""

        tags = " ".join(self.tags)
        return f"{self.title} {self.title} {tags} {tags} {self.content}"


@dataclass(frozen=True)
class SearchResult:
    """A relevant knowledge chunk and its deterministic retrieval score."""

    chunk: KnowledgeChunk
    score: float


def _tokenize(text: str) -> list[str]:
    return [
        TOKEN_NORMALIZATIONS.get(token, token)
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOP_WORDS
    ]


def _query_terms(query: str) -> list[str]:
    original_terms = _tokenize(query)
    expanded_terms = list(original_terms)
    for term in original_terms:
        expanded_terms.extend(sorted(QUERY_EXPANSIONS.get(term, set())))
    return expanded_terms


def load_knowledge_base(path: Path | str) -> list[KnowledgeChunk]:
    """Parse structured paragraph chunks from ``knowledge_base.txt``."""

    source = Path(path)
    raw_text = source.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", raw_text.strip())
    chunks: list[KnowledgeChunk] = []
    seen_ids: set[str] = set()

    for block_number, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 4:
            raise ValueError(
                f"Knowledge-base block {block_number} must contain an ID, "
                "Title, Tags, and content"
            )

        id_match = CHUNK_ID_PATTERN.fullmatch(lines[0])
        if not id_match:
            raise ValueError(
                f"Knowledge-base block {block_number} has an invalid chunk ID"
            )
        chunk_id = id_match.group(1)
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate knowledge-base chunk ID: {chunk_id}")

        if not lines[1].startswith("Title: "):
            raise ValueError(f"Chunk {chunk_id} is missing 'Title: '")
        if not lines[2].startswith("Tags: "):
            raise ValueError(f"Chunk {chunk_id} is missing 'Tags: '")

        title = lines[1].removeprefix("Title: ").strip()
        tags = tuple(
            tag.strip().lower()
            for tag in lines[2].removeprefix("Tags: ").split(",")
            if tag.strip()
        )
        content = " ".join(lines[3:]).strip()
        if not title or not tags or not content:
            raise ValueError(f"Chunk {chunk_id} contains an empty field")

        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                title=title,
                tags=tags,
                content=content,
            )
        )
        seen_ids.add(chunk_id)

    if not chunks:
        raise ValueError("The knowledge base does not contain any chunks")
    return chunks


def search_knowledge_base(
    query: str,
    *,
    path: Path | str,
    top_k: int = 5,
) -> list[SearchResult]:
    """Search the local knowledge base with BM25 plus a soft tag boost."""

    if not query.strip():
        return []
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")

    chunks = load_knowledge_base(path)
    query_terms = _query_terms(query)
    if not query_terms:
        return []

    documents = [_tokenize(chunk.searchable_text) for chunk in chunks]
    document_frequencies = Counter(
        term for terms in documents for term in set(terms)
    )
    average_length = sum(map(len, documents)) / len(documents)
    query_term_counts = Counter(query_terms)
    query_term_set = set(query_terms)
    specific_query_terms = query_term_set - GENERIC_QUERY_TERMS

    k1 = 1.5
    b = 0.75
    scored: list[SearchResult] = []

    for chunk, terms in zip(chunks, documents, strict=True):
        term_counts = Counter(terms)
        # A specific query concept must occur in the candidate. This prevents a
        # generic term such as "business trip" from making an unsupported pet
        # policy look relevant, while still allowing expanded synonyms such as
        # paperwork -> document and overseas -> international.
        if specific_query_terms and not specific_query_terms.intersection(
            term_counts
        ):
            continue
        score = 0.0

        for term, query_frequency in query_term_counts.items():
            term_frequency = term_counts[term]
            if term_frequency == 0:
                continue

            document_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1
                + (len(documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_normalization = term_frequency + k1 * (
                1 - b + b * len(terms) / average_length
            )
            score += (
                inverse_document_frequency
                * term_frequency
                * (k1 + 1)
                / length_normalization
                * min(query_frequency, 2)
            )

        matching_tags = query_term_set.intersection(chunk.tags)
        score += 0.35 * len(matching_tags)

        if score > 0:
            scored.append(SearchResult(chunk=chunk, score=score))

    return sorted(
        scored,
        key=lambda result: (-result.score, result.chunk.chunk_id),
    )[:top_k]


def format_search_results(query: str, results: list[SearchResult]) -> str:
    """Format raw evidence for the Retriever agent's tool output."""

    if not results:
        return (
            "RETRIEVAL STATUS: NOT_FOUND\n"
            f"QUERY: {query}\n"
            "No relevant snippets were found in the knowledge base."
        )

    snippets = [
        "RETRIEVAL STATUS: FOUND",
        f"QUERY: {query}",
    ]
    for result in results:
        snippets.extend(
            [
                "",
                f"[{result.chunk.chunk_id}] {result.chunk.title}",
                result.chunk.content,
            ]
        )
    return "\n".join(snippets)
