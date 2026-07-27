from pathlib import Path

import pytest

from src.config import ConfigurationError, Settings


KNOWLEDGE_BASE = Path(__file__).resolve().parents[1] / "knowledge_base.txt"


def test_builds_api_management_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://example.azure-api.net/llm",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", str(KNOWLEDGE_BASE))

    settings = Settings.from_env(env_file=None)

    assert (
        settings.api_base_url
        == "https://example.azure-api.net/llm/"
    )


def test_rejects_placeholder_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://example.azure-api.net/llm/",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "replace-with-your-api-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", str(KNOWLEDGE_BASE))

    with pytest.raises(ConfigurationError, match="put the provided key in .env"):
        Settings.from_env(env_file=None)


def test_reads_zero_wait_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://example.azure-api.net/llm/",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", str(KNOWLEDGE_BASE))
    monkeypatch.setenv("RATE_LIMIT_MAX_RETRIES", "0")
    monkeypatch.setenv("RATE_LIMIT_RETRY_SECONDS", "0")
    monkeypatch.setenv("DEMO_DELAY_SECONDS", "0")

    settings = Settings.from_env(env_file=None)

    assert settings.rate_limit_max_retries == 0
    assert settings.rate_limit_retry_seconds == 0
    assert settings.demo_delay_seconds == 0
