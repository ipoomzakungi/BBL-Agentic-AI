import os
import re

import pytest

from src.agents import answer_question
from src.config import Settings


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_TESTS") != "1",
        reason="Set RUN_LIVE_TESTS=1 to consume API quota.",
    ),
]


def test_international_travel_answer_is_grounded_and_proofread() -> None:
    answer = answer_question(
        "What is the policy on international travel?",
        Settings.from_env(),
    )
    normalized = re.sub(r"\s+", " ", answer.lower())

    assert "[travel-001]" in normalized
    assert "30 calendar days" in normalized
    assert "manager" in normalized
    assert "department head" in normalized
    assert "byyour" not in normalized
