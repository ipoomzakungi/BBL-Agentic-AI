"""Command-line entry point for the two-agent RAG challenge."""

from __future__ import annotations

import argparse
import sys
import time

from .agents import answer_question
from .config import ConfigurationError, Settings


DEMO_QUESTIONS = (
    "What is the policy on international travel?",
    "How long do I have to submit my expenses after returning?",
    "What approvals and flight class rules apply to an overseas trip?",
    "Can employees bring pets on business trips?",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask the two-agent local knowledge-base system a question."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to answer. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run four evaluation queries suitable for screenshots.",
    )
    parser.add_argument(
        "--demo-delay",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Wait between demo queries to respect token-per-minute limits "
            "(default: DEMO_DELAY_SECONDS)."
        ),
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show effective non-secret configuration and exit.",
    )
    return parser


def _run_questions(
    questions: list[str],
    settings: Settings,
    *,
    delay_seconds: float = 0.0,
) -> None:
    for index, question in enumerate(questions, start=1):
        if len(questions) > 1:
            print(f"\n{'=' * 72}\nQuery {index}: {question}\n{'-' * 72}")
        print(answer_question(question, settings))
        if index < len(questions) and delay_seconds > 0:
            print(
                f"\nWaiting {delay_seconds:g}s for the API token window...",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        parser.error(str(exc))

    if args.show_config:
        print(f"API endpoint: {settings.api_base_url}")
        print(f"Model: {settings.azure_deployment}")
        print(f"Rate-limit retries: {settings.rate_limit_max_retries}")
        print(
            "Rate-limit fallback delay: "
            f"{settings.rate_limit_retry_seconds:g}s"
        )
        print(f"Demo delay: {settings.demo_delay_seconds:g}s")
        return 0

    if args.demo:
        if args.demo_delay is not None and args.demo_delay < 0:
            parser.error("--demo-delay must not be negative")
        questions = list(DEMO_QUESTIONS)
        delay_seconds = (
            args.demo_delay
            if args.demo_delay is not None
            else settings.demo_delay_seconds
        )
    elif args.question:
        questions = [" ".join(args.question)]
        delay_seconds = 0.0
    else:
        question = input("Ask a question: ").strip()
        if not question:
            print("A question is required.", file=sys.stderr)
            return 2
        questions = [question]
        delay_seconds = 0.0

    _run_questions(
        questions,
        settings,
        delay_seconds=delay_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
