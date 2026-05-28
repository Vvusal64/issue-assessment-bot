"""Configuration loading.

Reads from environment variables (and an optional .env file) and validates
that the values needed for the chosen run mode are present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # python-dotenv is optional at runtime; .env loading is best-effort
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore[no-redef]
        return False


DEFAULT_JQL = (
    'type = "OEM App Bug" AND status = Open '
    'ORDER BY created DESC'
)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    # Jira
    jira_base_url: str
    jira_email: str
    jira_api_token: str
    jira_jql: str

    # Slack
    slack_bot_token: str

    # State
    state_file: Path

    # Claude fallback
    anthropic_api_key: Optional[str]
    anthropic_model: str

    # Admin Portal verification (optional; chained ahead of the local
    # JSON resolver when both URL and token are set).
    admin_portal_url: Optional[str]
    admin_portal_token: Optional[str]

    # Misc
    log_level: str

    @classmethod
    def from_env(cls, *, env_file: Optional[Path] = None) -> "Config":
        if env_file and env_file.exists():
            load_dotenv(env_file)
        else:
            # Best-effort load of a .env in the cwd
            load_dotenv()

        def _req(name: str) -> str:
            value = os.environ.get(name, "").strip()
            if not value:
                raise ConfigError(
                    f"Required environment variable {name} is missing. "
                    f"See .env.example for the full list."
                )
            return value

        jira_base_url = _req("JIRA_BASE_URL").rstrip("/")
        jira_email = _req("JIRA_EMAIL")
        jira_api_token = _req("JIRA_API_TOKEN")
        slack_bot_token = _req("SLACK_BOT_TOKEN")
        if not slack_bot_token.startswith("xoxb-"):
            raise ConfigError(
                "SLACK_BOT_TOKEN must be a bot token starting with 'xoxb-'."
            )

        state_file = Path(
            os.environ.get("STATE_FILE", "state.json")
        ).expanduser().resolve()

        return cls(
            jira_base_url=jira_base_url,
            jira_email=jira_email,
            jira_api_token=jira_api_token,
            jira_jql=os.environ.get("JIRA_JQL", DEFAULT_JQL).strip(),
            slack_bot_token=slack_bot_token,
            state_file=state_file,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            anthropic_model=os.environ.get(
                "ANTHROPIC_MODEL", "claude-sonnet-4-5"
            ),
            admin_portal_url=(os.environ.get("ADMIN_PORTAL_URL") or "").strip()
                or None,
            admin_portal_token=(os.environ.get("ADMIN_PORTAL_TOKEN") or "").strip()
                or None,
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        )
