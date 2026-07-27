"""Agent definitions, low-token orchestration, and rate-limit handling."""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable

from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI, Omit, RateLimitError

from .config import Settings
from .retrieval import (
    format_search_results,
    load_knowledge_base,
    search_knowledge_base,
)


def _build_model_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.azure_api_key,
        base_url=settings.api_base_url,
        # Retry at the agent-stage level so a failed Report Generator request
        # does not rerun the already successful Retriever stage.
        max_retries=0,
    )


def _api_management_headers(settings: Settings) -> dict[str, str | Omit]:
    # Omit is the SDK-supported way to suppress its default Bearer header for a
    # request. The candidate gateway authenticates with api-key instead.
    return {
        "Authorization": Omit(),
        "api-key": settings.azure_api_key,
    }


def _build_search_tool(settings: Settings):
    @function_tool
    def search_local_knowledge(query: str) -> str:
        """Search the local policy knowledge base for relevant raw snippets.

        Args:
            query: A concise search query preserving important names and numbers.
        """

        results = search_knowledge_base(
            query,
            path=settings.knowledge_base_path,
            top_k=3,
        )
        return format_search_results(query, results)

    return search_local_knowledge


def build_agents(settings: Settings) -> tuple[Agent, Agent]:
    """Build the Retriever and Report Generator for sequential execution."""

    client = _build_model_client(settings)
    model = OpenAIResponsesModel(
        model=settings.azure_deployment,
        openai_client=client,
    )
    api_headers = _api_management_headers(settings)

    # The provided Azure key is for model inference, not OpenAI platform tracing.
    set_tracing_disabled(True)

    # Validate the file at startup instead of failing only after an API call.
    load_knowledge_base(settings.knowledge_base_path)

    retriever = Agent(
        name="Data Retriever",
        instructions=(
            "Retrieve evidence only. Call search_local_knowledge exactly once "
            "with a concise query that preserves important names and numbers. "
            "Return its output unchanged; never answer or summarize."
        ),
        tools=[_build_search_tool(settings)],
        model=model,
        model_settings=ModelSettings(
            tool_choice="required",
            parallel_tool_calls=False,
            extra_headers=api_headers,
            max_tokens=120,
            reasoning={"effort": "minimal"},
        ),
        # This makes the first search result the Retriever's final output,
        # enforcing that this agent retrieves evidence but never answers.
        tool_use_behavior="stop_on_first_tool",
    )

    report_generator = Agent(
        name="Report Generator",
        instructions=(
            "Answer the question using only the supplied evidence. Be concise, "
            "combine facts without repetition, and cite chunk IDs such as "
            "[TRAVEL-001]. If evidence says NOT_FOUND, state that the knowledge "
            "base lacks the information. Proofread grammar and spacing. Never "
            "add outside facts or mention internal orchestration."
        ),
        model=model,
        model_settings=ModelSettings(
            extra_headers=api_headers,
            max_tokens=280,
            reasoning={"effort": "minimal"},
        ),
    )
    return retriever, report_generator


def _parse_reset_duration(value: str | None) -> float | None:
    """Parse rate-limit durations such as ``60``, ``1m``, or ``250ms``."""

    if not value:
        return None
    value = value.strip().lower()

    try:
        numeric = float(value)
        return numeric if numeric >= 0 else None
    except ValueError:
        pass

    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m)", value)
    if not matches:
        return None

    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0}
    return sum(float(amount) * multipliers[unit] for amount, unit in matches)


def _rate_limit_delay(error: RateLimitError, fallback: float) -> float:
    headers = error.response.headers

    retry_after_ms = _parse_reset_duration(headers.get("retry-after-ms"))
    if retry_after_ms is not None:
        return max(retry_after_ms / 1000.0, 0.0)

    for header_name in ("retry-after", "x-ratelimit-reset-tokens"):
        parsed = _parse_reset_duration(headers.get(header_name))
        if parsed is not None:
            return max(parsed, 0.0)

    return fallback


def _run_agent_with_rate_limit_retry(
    agent: Agent,
    input_text: str,
    settings: Settings,
    *,
    stage_name: str,
    max_turns: int,
    sleep_fn: Callable[[float], None] = time.sleep,
):
    """Run one agent stage and retry only that stage after an HTTP 429."""

    for attempt in range(settings.rate_limit_max_retries + 1):
        try:
            return Runner.run_sync(
                agent,
                input_text,
                max_turns=max_turns,
            )
        except RateLimitError as error:
            if attempt >= settings.rate_limit_max_retries:
                raise

            delay = _rate_limit_delay(
                error,
                settings.rate_limit_retry_seconds,
            )
            print(
                f"Rate limit reached during {stage_name}. Retrying in "
                f"{delay:g}s ({attempt + 1}/"
                f"{settings.rate_limit_max_retries})...",
                file=sys.stderr,
            )
            sleep_fn(delay)


def answer_question(question: str, settings: Settings) -> str:
    """Run the sequential two-agent workflow for one user question."""

    if not question.strip():
        raise ValueError("Question must not be empty")

    cleaned_question = question.strip()
    retriever, report_generator = build_agents(settings)

    retrieval_result = _run_agent_with_rate_limit_retry(
        retriever,
        cleaned_question,
        settings,
        stage_name="Data Retriever",
        max_turns=3,
    )
    evidence = str(retrieval_result.final_output)
    if not evidence.strip():
        raise RuntimeError("Data Retriever returned empty evidence")

    report_input = (
        f"QUESTION:\n{cleaned_question}\n\n"
        f"RETRIEVED EVIDENCE:\n{evidence}"
    )
    report_result = _run_agent_with_rate_limit_retry(
        report_generator,
        report_input,
        settings,
        stage_name="Report Generator",
        max_turns=2,
    )
    final_answer = str(report_result.final_output).strip()
    if not final_answer:
        raise RuntimeError("Report Generator returned an empty answer")
    return final_answer
