from pathlib import Path

import pytest

from src.retrieval import (
    format_search_results,
    load_knowledge_base,
    search_knowledge_base,
)


KNOWLEDGE_BASE = Path(__file__).resolve().parents[1] / "knowledge_base.txt"


def test_loads_unique_structured_chunks() -> None:
    chunks = load_knowledge_base(KNOWLEDGE_BASE)

    assert len(chunks) == 8
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert chunks[0].chunk_id == "TRAVEL-001"


@pytest.mark.parametrize(
    ("query", "expected_chunk_id"),
    [
        ("What is the policy on international travel?", "TRAVEL-001"),
        (
            "How long do I have to submit my expenses after returning?",
            "EXPENSE-001",
        ),
        ("What paperwork is needed before going abroad?", "TRAVEL-002"),
        ("Can I claim hotel and visa costs?", "EXPENSE-002"),
    ],
)
def test_retrieves_expected_top_chunk(
    query: str,
    expected_chunk_id: str,
) -> None:
    results = search_knowledge_base(query, path=KNOWLEDGE_BASE, top_k=3)

    assert results
    assert results[0].chunk.chunk_id == expected_chunk_id


def test_multi_topic_query_retrieves_approval_and_flight_rules() -> None:
    results = search_knowledge_base(
        "What approvals and flight class rules apply to an overseas trip?",
        path=KNOWLEDGE_BASE,
        top_k=5,
    )
    chunk_ids = {result.chunk.chunk_id for result in results}

    assert {"TRAVEL-001", "TRAVEL-003"}.issubset(chunk_ids)


def test_unknown_topic_returns_no_evidence() -> None:
    results = search_knowledge_base(
        "Can employees bring pets on business trips?",
        path=KNOWLEDGE_BASE,
    )

    assert results == []
    assert "NOT_FOUND" in format_search_results("pets", results)


def test_top_k_is_validated() -> None:
    with pytest.raises(ValueError, match="between 1 and 10"):
        search_knowledge_base("travel", path=KNOWLEDGE_BASE, top_k=0)
