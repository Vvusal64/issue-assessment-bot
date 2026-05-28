"""Slack Bot API client.

Resolves a Jira assignee email to a Slack user ID, opens a direct-message
channel, and posts a structured notification with the suggested
metadata. All transport errors are retried; semantic errors raised by
Slack (e.g. ``users_not_found``) are wrapped in :class:`SlackError` so
the orchestrator can handle them per-issue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from .jira_client import RelatedIssue
from .utils import retry


log = logging.getLogger("bot.slack")


class SlackError(RuntimeError):
    """Raised when Slack returns ``ok=false`` or a non-2xx response."""

    def __init__(self, method: str, error: str, *, response: Optional[Dict[str, Any]] = None):
        super().__init__(f"Slack {method} failed: {error}")
        self.method = method
        self.error = error
        self.response = response or {}


@dataclass
class SlackPostResult:
    channel: str
    ts: str


class SlackClient:
    """Minimal Slack web-API client."""

    BASE_URL = "https://slack.com/api"

    def __init__(self, bot_token: str,
                 *, session: Optional[requests.Session] = None,
                 timeout: float = 15.0) -> None:
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {bot_token}",
            "User-Agent": "jira-assessment-bot/1.0",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup_user_by_email(self, email: str) -> str:
        """Return the Slack user ID for ``email`` or raise SlackError."""
        data = self._call(
            "users.lookupByEmail", method="GET", params={"email": email},
        )
        return data["user"]["id"]

    def open_dm(self, user_id: str) -> str:
        """Open an IM channel and return its channel ID."""
        data = self._call(
            "conversations.open", method="POST", json={"users": user_id},
        )
        return data["channel"]["id"]

    def post_message(
        self, *, channel: str, text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> SlackPostResult:
        payload: Dict[str, Any] = {"channel": channel, "text": text}
        if blocks:
            payload["blocks"] = blocks
        data = self._call("chat.postMessage", method="POST", json=payload)
        return SlackPostResult(
            channel=data.get("channel", channel),
            ts=data.get("ts", ""),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @retry(logger=log)
    def _http(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        return self._session.request(method, url, timeout=self.timeout, **kwargs)

    def _call(self, api_method: str, *, method: str = "POST",
              params: Optional[Dict[str, Any]] = None,
              json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{api_method}"
        if method.upper() == "GET":
            resp = self._http("GET", url, params=params)
        else:
            # chat.postMessage uses application/json
            headers = {"Content-Type": "application/json; charset=utf-8"}
            resp = self._http("POST", url, json=json, headers=headers)
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            raise SlackError(api_method, f"non-JSON response: {resp.text[:200]}")

        if not resp.ok or not data.get("ok"):
            err = data.get("error") or f"http_{resp.status_code}"
            raise SlackError(api_method, err, response=data)
        return data


# ----------------------------------------------------------------------
# Message rendering
# ----------------------------------------------------------------------

def compute_status_line(
    *, app_name: Optional[str], app_version: Optional[str],
    package_name: Optional[str],
    confidence: Optional[str] = None,
    admin_portal_status: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(emoji, status_text)`` for the top of the message.

    Decision table (admin_portal_status × confidence × completeness):

    * **✅ Metadata detected with high confidence (Admin Portal verified)**
      — all three fields extracted, confidence is ``"high"`` AND the
      package was independently confirmed against the Admin Portal.
      The user's instruction is explicit: the green banner is only
      earned when the portal corroborates the bot's answer.
    * **⚠️ Metadata partially detected — manual validation required**
      — anything in between. Includes: partial fields; all three
      fields but confidence < high; all three fields with high
      confidence but the Admin Portal could not verify the package
      (``no_match`` or ``version_mismatch``); or fields extracted
      but the portal was never consulted (``not_checked``) — we
      still ask for manual confirmation rather than ship a green
      banner the user can't trace back to the portal.
    * **❌ Metadata not detected — manual validation required** — no
      fields extracted at all.
    """
    have_pkg = bool(package_name)
    have_name = bool(app_name)
    have_ver = bool(app_version)
    confidence_norm = (confidence or "").strip().lower()
    portal_norm = (admin_portal_status or "").strip().lower()
    if (have_pkg and have_name and have_ver
            and confidence_norm == "high"
            and portal_norm == "verified"):
        return ("✅",  # ✅
                "Metadata detected with high confidence "
                "(verified against Admin Portal)")
    if not have_pkg and not have_name and not have_ver:
        return ("❌",  # ❌
                "Metadata not detected — manual validation required")
    return ("⚠️",  # ⚠️
            "Metadata partially detected — manual validation required")


def _format_admin_portal_line(
    status: Optional[str],
    portal_versions: Optional[List[str]] = None,
) -> str:
    """One-line summary of the Admin Portal cross-check outcome."""
    s = (status or "").strip().lower()
    if s == "verified":
        if portal_versions:
            joined = ", ".join(portal_versions[:4])
            extra = "" if len(portal_versions) <= 4 else f" (+{len(portal_versions) - 4} more)"
            return f"✓ Verified — published versions: {joined}{extra}"
        return "✓ Verified"
    if s == "version_mismatch":
        if portal_versions:
            joined = ", ".join(portal_versions[:6])
            extra = (
                "" if len(portal_versions) <= 6
                else f" (+{len(portal_versions) - 6} more)"
            )
            return (
                "⚠ Package found in Admin Portal but the bot could not "
                "confirm the version against the published list. "
                f"Approved versions: {joined}{extra}. "
                "Please verify manually before reporting."
            )
        return (
            "⚠ Package found in Admin Portal but the version could not "
            "be confirmed. Manual check required."
        )
    if s == "no_match":
        return (
            "✗ Package NOT found in Admin Portal — manual investigation "
            "required."
        )
    return "— not checked"


def render_message(
    *, issue_key: str, issue_url: str, assignee_name: str,
    summary: str,
    app_name: Optional[str], app_version: Optional[str],
    package_name: Optional[str], confidence: str,
    version_code: Optional[str] = None,
    environment_name: Optional[str] = None,
    related_issues: Optional[List[RelatedIssue]] = None,
    note: Optional[str] = None,
    admin_portal_status: Optional[str] = None,
    admin_portal_versions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build (text, blocks) for the Slack notification.

    Layout (top to bottom):
      1. Status line (✅ / ⚠️ / ❌) — at-a-glance signal for the assignee.
         ✅ requires Admin Portal verification, not just high confidence.
      2. "New Issue Assessment Insight" header.
      3. Issue key + Open in Jira link + Assignee.
      4. Summary.
      5. Detected metadata (app name / version / package / confidence /
         Admin Portal cross-check outcome).
      6. Related ongoing tickets (only included when ``related_issues``
         is not ``None``; "None found" rendered for an empty list).
      7. Optional context note.

    ``admin_portal_status`` is one of: ``"verified"``,
    ``"version_mismatch"``, ``"no_match"``, ``"not_checked"`` (or
    ``None`` — treated as not_checked).
    """
    def _val(v: Optional[str]) -> str:
        return v if v else "not detected"

    status_emoji, status_text = compute_status_line(
        app_name=app_name,
        app_version=app_version,
        package_name=package_name,
        confidence=confidence,
        admin_portal_status=admin_portal_status,
    )

    admin_line = _format_admin_portal_line(
        admin_portal_status, admin_portal_versions,
    )

    text_lines: List[str] = [
        f"{status_emoji} {status_text}",
        "",
        "New Issue Assessment Insight",
        "",
        f"Issue: {issue_key}",
        f"Open in Jira: {issue_url}",
        f"Assignee: {assignee_name}",
        "",
        "Summary:",
        summary,
        "",
        "Detected metadata:",
        f"  • App Name: {_val(app_name)}",
        f"  • App Version: {_val(app_version)}",
        f"  • Version Code: {_val(version_code)}",
        f"  • Package Name: {_val(package_name)}",
        f"  • Environment: {_val(environment_name)}",
        f"  • Confidence: {confidence.title()}",
        f"  • Admin Portal: {admin_line}",
    ]
    if related_issues is not None:
        text_lines.extend(["", "Related ongoing tickets:"])
        if related_issues:
            for r in related_issues:
                text_lines.append(f"  • {r.key} — {r.summary}")
        else:
            text_lines.append("  None found")
    if note:
        text_lines.extend(["", f"Note: {note}"])

    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{status_emoji} *{status_text}*",
            },
        },
        {"type": "divider"},
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "New Issue Assessment Insight",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn",
                 "text": f"*Issue*\n<{issue_url}|{issue_key}>"},
                {"type": "mrkdwn",
                 "text": f"*Assignee*\n{assignee_name}"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Open in Jira*\n<{issue_url}|{issue_url}>",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Summary*\n{summary[:500]}",
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Detected metadata*"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn",
                 "text": f"*App Name*\n{_val(app_name)}"},
                {"type": "mrkdwn",
                 "text": f"*App Version*\n{_val(app_version)}"},
                {"type": "mrkdwn",
                 "text": f"*Version Code*\n{_val(version_code)}"},
                {"type": "mrkdwn",
                 "text": f"*Package Name*\n`{_val(package_name)}`"},
                {"type": "mrkdwn",
                 "text": f"*Environment*\n{_val(environment_name)}"},
                {"type": "mrkdwn",
                 "text": f"*Confidence*\n{confidence.title()}"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Admin Portal*\n{admin_line}",
            },
        },
    ]
    if related_issues is not None:
        if related_issues:
            related_md = "\n".join(
                f"• <{r.url}|{r.key}> — {_truncate(r.summary, 140)}"
                for r in related_issues
            )
        else:
            related_md = "_None found_"
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Related ongoing tickets*\n{related_md}",
            },
        })
    if note:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f":memo: {note}"}],
        })

    return {"text": "\n".join(text_lines), "blocks": blocks}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
