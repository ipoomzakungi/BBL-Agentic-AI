"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base.txt"
DEFAULT_RATE_LIMIT_MAX_RETRIES = 2
DEFAULT_RATE_LIMIT_RETRY_SECONDS = 30.0
DEFAULT_DEMO_DELAY_SECONDS = 60.0


class ConfigurationError(ValueError):
    """Raised when required application configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the Azure OpenAI connection and local knowledge base."""

    azure_endpoint: str
    azure_api_key: str
    azure_deployment: str
    knowledge_base_path: Path = DEFAULT_KNOWLEDGE_BASE
    rate_limit_max_retries: int = DEFAULT_RATE_LIMIT_MAX_RETRIES
    rate_limit_retry_seconds: float = DEFAULT_RATE_LIMIT_RETRY_SECONDS
    demo_delay_seconds: float = DEFAULT_DEMO_DELAY_SECONDS

    @property
    def api_base_url(self) -> str:
        """Return the API Management base URL expected by the OpenAI client."""

        return f"{self.azure_endpoint.rstrip('/')}/"

    @classmethod
    def from_env(
        cls,
        *,
        env_file: Path | str | None = PROJECT_ROOT / ".env",
        require_api_key: bool = True,
    ) -> "Settings":
        """Load settings from a .env file and the process environment."""

        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        retries_value = os.getenv(
            "RATE_LIMIT_MAX_RETRIES",
            str(DEFAULT_RATE_LIMIT_MAX_RETRIES),
        ).strip()
        retry_seconds_value = os.getenv(
            "RATE_LIMIT_RETRY_SECONDS",
            str(DEFAULT_RATE_LIMIT_RETRY_SECONDS),
        ).strip()
        demo_delay_value = os.getenv(
            "DEMO_DELAY_SECONDS",
            str(DEFAULT_DEMO_DELAY_SECONDS),
        ).strip()
        kb_value = os.getenv("KNOWLEDGE_BASE_PATH", "").strip()
        knowledge_base_path = (
            Path(kb_value).expanduser().resolve()
            if kb_value
            else DEFAULT_KNOWLEDGE_BASE
        )

        errors: list[str] = []
        if not endpoint:
            errors.append("AZURE_OPENAI_ENDPOINT is required")
        elif not endpoint.startswith("https://"):
            errors.append("AZURE_OPENAI_ENDPOINT must start with https://")

        if not deployment:
            errors.append("AZURE_OPENAI_DEPLOYMENT is required")

        try:
            rate_limit_max_retries = int(retries_value)
            if rate_limit_max_retries < 0:
                raise ValueError
        except ValueError:
            errors.append("RATE_LIMIT_MAX_RETRIES must be a non-negative integer")
            rate_limit_max_retries = DEFAULT_RATE_LIMIT_MAX_RETRIES

        try:
            rate_limit_retry_seconds = float(retry_seconds_value)
            if rate_limit_retry_seconds < 0:
                raise ValueError
        except ValueError:
            errors.append("RATE_LIMIT_RETRY_SECONDS must be a non-negative number")
            rate_limit_retry_seconds = DEFAULT_RATE_LIMIT_RETRY_SECONDS

        try:
            demo_delay_seconds = float(demo_delay_value)
            if demo_delay_seconds < 0:
                raise ValueError
        except ValueError:
            errors.append("DEMO_DELAY_SECONDS must be a non-negative number")
            demo_delay_seconds = DEFAULT_DEMO_DELAY_SECONDS

        placeholder_values = {
            "",
            "replace-with-your-api-key",
            "your-api-key-here",
        }
        if require_api_key and api_key.lower() in placeholder_values:
            errors.append(
                "AZURE_OPENAI_API_KEY is missing; put the provided key in .env"
            )

        if not knowledge_base_path.is_file():
            errors.append(
                f"Knowledge base file was not found: {knowledge_base_path}"
            )

        if errors:
            raise ConfigurationError("; ".join(errors))

        return cls(
            azure_endpoint=endpoint,
            azure_api_key=api_key,
            azure_deployment=deployment,
            knowledge_base_path=knowledge_base_path,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_retry_seconds=rate_limit_retry_seconds,
            demo_delay_seconds=demo_delay_seconds,
        )
