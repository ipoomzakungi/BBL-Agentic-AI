from types import SimpleNamespace

import httpx
import pytest
from agents import Agent
from openai import RateLimitError

import src.agents as workflow
from src.config import DEFAULT_KNOWLEDGE_BASE, Settings


def _settings(**overrides) -> Settings:
    values = {
        "azure_endpoint": "https://example.azure-api.net/llm/",
        "azure_api_key": "test-secret",
        "azure_deployment": "gpt-5-mini",
        "knowledge_base_path": DEFAULT_KNOWLEDGE_BASE,
        "rate_limit_max_retries": 2,
        "rate_limit_retry_seconds": 60.0,
    }
    values.update(overrides)
    return Settings(**values)


def test_answer_question_passes_retriever_output_to_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = Agent(name="Data Retriever")
    reporter = Agent(name="Report Generator")
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        workflow,
        "build_agents",
        lambda settings: (retriever, reporter),
    )

    def fake_run(agent, input_text, settings, **kwargs):
        calls.append((agent.name, input_text))
        if agent is retriever:
            return SimpleNamespace(
                final_output=(
                    "[TRAVEL-001] International Travel Approval\n"
                    "Submit at least 30 calendar days before departure."
                )
            )
        return SimpleNamespace(final_output="Grounded answer [TRAVEL-001].")

    monkeypatch.setattr(
        workflow,
        "_run_agent_with_rate_limit_retry",
        fake_run,
    )

    answer = workflow.answer_question(
        "What is the international travel policy?",
        _settings(),
    )

    assert answer == "Grounded answer [TRAVEL-001]."
    assert [name for name, _ in calls] == [
        "Data Retriever",
        "Report Generator",
    ]
    assert "RETRIEVED EVIDENCE:" in calls[1][1]
    assert "[TRAVEL-001]" in calls[1][1]


def test_answer_question_rejects_empty_report_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = Agent(name="Data Retriever")
    reporter = Agent(name="Report Generator")

    monkeypatch.setattr(
        workflow,
        "build_agents",
        lambda settings: (retriever, reporter),
    )

    def fake_run(agent, input_text, settings, **kwargs):
        output = "[TRAVEL-001] Evidence" if agent is retriever else ""
        return SimpleNamespace(final_output=output)

    monkeypatch.setattr(
        workflow,
        "_run_agent_with_rate_limit_retry",
        fake_run,
    )

    with pytest.raises(RuntimeError, match="empty answer"):
        workflow.answer_question("Question", _settings())


def test_retries_only_the_rate_limited_agent_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []
    response = httpx.Response(
        status_code=429,
        headers={"retry-after": "2.5"},
        request=httpx.Request(
            "POST",
            "https://example.azure-api.net/llm/responses",
        ),
    )

    def fake_run_sync(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError(
                "Token limit reached",
                response=response,
                body=None,
            )
        return SimpleNamespace(final_output="success")

    monkeypatch.setattr(workflow.Runner, "run_sync", fake_run_sync)

    result = workflow._run_agent_with_rate_limit_retry(
        Agent(name="Report Generator"),
        "input",
        _settings(),
        stage_name="Report Generator",
        max_turns=2,
        sleep_fn=delays.append,
    )

    assert result.final_output == "success"
    assert attempts == 2
    assert delays == [2.5]


@pytest.mark.parametrize(
    ("header_value", "expected_seconds"),
    [
        ("60", 60.0),
        ("1m", 60.0),
        ("1m 5s", 65.0),
        ("250ms", 0.25),
    ],
)
def test_parses_rate_limit_reset_duration(
    header_value: str,
    expected_seconds: float,
) -> None:
    assert workflow._parse_reset_duration(header_value) == expected_seconds
