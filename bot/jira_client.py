"""Atlassian Jira Cloud REST client.

Authenticates with HTTP Basic (email + API token), runs the Issue
Assessment JQL, and returns issues enriched with comments and the text
content of small text-based attachments.

Implements explicit, narrow methods rather than a generic wrapper so the
call sites in :mod:`bot.main` stay easy to read.
"""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import requests
from requests.auth import HTTPBasicAuth

from .utils import adf_to_text, retry


log = logging.getLogger("bot.jira")


# Attachment policy: we'd rather have the first chunk of a large
# log than nothing at all. The cap below is a streaming truncation
# limit, not an upfront skip — we no longer drop files purely on
# declared size.
import os as _os
MAX_ATTACHMENT_BYTES = int(
    _os.environ.get("MAX_ATTACHMENT_BYTES", str(50 * 1024 * 1024))
)
# Separate, higher cap for zip downloads — extraction inside is
# bounded too. Large vendor traces (500MB+) still get skipped because
# downloading them is prohibitively expensive.
MAX_ZIP_BYTES = int(
    _os.environ.get("MAX_ZIP_BYTES", str(80 * 1024 * 1024))
)
TEXT_LIKE_EXTENSIONS = {
    ".txt", ".log", ".logcat", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".ini", ".cfg", ".conf", ".md", ".html", ".htm", ".properties",
    ".gradle", ".kts", ".manifest",
}
TEXT_LIKE_MIME_PREFIXES = ("text/", "application/json", "application/xml")


@dataclass
class Attachment:
    filename: str
    mime_type: str
    size: int
    content_url: str
    text: Optional[str] = None  # populated for text-like attachments


@dataclass
class Comment:
    author: str
    body: str            # plain text (ADF-flattened)
    created: str


@dataclass
class StructuredFields:
    """Parsed values from Jira custom fields.

    These come from the right-side Details panel of a Jira ticket and
    are the single most trustworthy source of app metadata — they were
    set deliberately by a human (or by an automation that knows the
    ticket better than we do). Anything we read here MUST override
    weaker regex extraction in the merge step.
    """
    app_name: Optional[str] = None
    package_name: Optional[str] = None
    version_names: List[str] = field(default_factory=list)
    app_fix_versions: List[str] = field(default_factory=list)
    version_code: Optional[str] = None
    environment_name: Optional[str] = None

    def has_any(self) -> bool:
        return bool(
            self.app_name or self.package_name
            or self.version_names or self.app_fix_versions
            or self.version_code or self.environment_name
        )


@dataclass
class RelatedIssue:
    """Lightweight projection of a Jira issue used for the
    "Related ongoing tickets" section of the Slack notification.
    Intentionally minimal — we only fetch ``summary`` and ``status``
    to keep the lookup cheap and side-effect-free.
    """
    key: str
    summary: str
    url: str
    status: str = ""


def _jql_escape_string(value: str) -> str:
    """Escape a value for use inside a JQL double-quoted string.

    JQL string literals interpret backslash and double-quote, so those
    are the only two characters we need to escape.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_related_open_jql(
    *, exclude_key: str,
    package_name: Optional[str],
    app_name: Optional[str] = None,  # accepted but deliberately ignored
) -> Optional[str]:
    """Build the JQL for the Related-ongoing-tickets lookup.

    Strictly **package-only**. Returns ``None`` if no package is
    given — the user requirement is that related tickets must share
    the same package coordinate, not just the same app name. A
    name-only fallback would surface tickets about a different
    variant of the same app family (e.g. Stellantis Zoom vs Forvia
    Zoom for Cars), which is exactly the false-positive class we want
    to avoid.

    The ``app_name`` argument is accepted for source-compatibility
    with older callers but no longer affects the output.
    """
    if not (package_name and package_name.strip()):
        return None
    clause = f'text ~ "{_jql_escape_string(package_name.strip())}"'
    return (
        f'type = "OEM App Bug" '
        f'AND resolution = Unresolved '
        f'AND key != "{_jql_escape_string(exclude_key)}" '
        f'AND ({clause}) '
        f'ORDER BY updated DESC'
    )


@dataclass
class JiraIssue:
    key: str
    summary: str
    description: str                  # plain text (ADF-flattened)
    status: str
    issue_type: str
    assignee_display_name: Optional[str]
    assignee_email: Optional[str]
    assignee_account_id: Optional[str]
    created: str
    updated: str
    comments: List[Comment] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)
    # Parsed from custom fields. ``None`` means "we didn't read structured
    # fields for this issue" (e.g. legacy fixture); an empty
    # ``StructuredFields()`` means "we read them and they were empty".
    structured: Optional[StructuredFields] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        # Filled in by the client when constructing the issue.
        return self.raw.get("_self_url", "")


class JiraClient:
    """Thin wrapper around the Jira Cloud REST API v3."""

    def __init__(self, base_url: str, email: str, api_token: str,
                 *, session: Optional[requests.Session] = None,
                 timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.auth = HTTPBasicAuth(email, api_token)
        self._session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "jira-assessment-bot/1.0",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_assigned_open_issues(
        self, jql: str, *, page_size: int = 50,
        max_results: Optional[int] = None,
    ) -> List[JiraIssue]:
        """Return all issues matching ``jql`` that have an assignee.

        Iterates through pages until either the result set is exhausted
        or ``max_results`` is reached.
        """
        keys = [k for k in self._search_keys(jql, page_size=page_size)]
        if max_results is not None:
            keys = keys[:max_results]

        log.info("Jira search returned %d issue key(s)", len(keys))

        issues: List[JiraIssue] = []
        for key in keys:
            issue = self.get_issue(key)
            if issue.assignee_account_id is None:
                log.debug("Skipping %s: no assignee", key)
                continue
            issues.append(issue)
        return issues

    def get_issue(self, key: str) -> JiraIssue:
        """Fetch a single issue with all fields (system + custom).

        ``fields=*all`` ensures the structured custom fields shown in
        the Jira Details panel (App Name, Package Name, Version Name,
        App Version, App Fix Version, …) come back in the response so
        the bot can prefer them over regex extraction. ``expand=names``
        adds a top-level ``names`` map for human-readable field names.
        """
        params = {
            "fields": "*all",
            "expand": "renderedFields,names",
        }
        url = f"{self.base_url}/rest/api/3/issue/{key}"
        resp = self._get(url, params=params)
        if resp.status_code == 404:
            raise LookupError(f"Issue {key} not found")
        resp.raise_for_status()
        data = resp.json()
        return self._build_issue(data)

    def list_fields(self) -> List[Dict[str, Any]]:
        """List every Jira field (system + custom).

        Used once at startup by :class:`bot.jira_fields.StructuredFieldReader`
        to translate display names like "App Name" into custom field
        IDs like ``customfield_10100``. The result is cached in-memory
        on the client so subsequent calls are free.
        """
        if getattr(self, "_field_cache", None) is not None:
            return self._field_cache  # type: ignore[return-value]
        url = f"{self.base_url}/rest/api/3/field"
        resp = self._get(url)
        resp.raise_for_status()
        data = resp.json()
        self._field_cache = data if isinstance(data, list) else []
        return self._field_cache

    def find_related_open_issues(
        self, *, exclude_key: str,
        package_name: Optional[str] = None,
        app_name: Optional[str] = None,
        max_results: int = 5,
    ) -> List[RelatedIssue]:
        """Return up to ``max_results`` open OEM App Bug issues that
        appear to discuss the same app.

        See :func:`build_related_open_jql` for the search semantics.
        Returns ``[]`` when no useful signal is available — never
        raises in that case so callers can use this as a non-blocking
        enrichment step.
        """
        jql = build_related_open_jql(
            exclude_key=exclude_key,
            package_name=package_name,
            app_name=app_name,
        )
        if jql is None:
            return []
        return self._search_lightweight(jql, max_results=max_results)

    def _search_lightweight(
        self, jql: str, *, max_results: int,
    ) -> List[RelatedIssue]:
        """Run a JQL search and return key+summary+status only.

        Uses the modern ``/search/jql`` endpoint and falls back to the
        legacy ``/search`` endpoint if the modern one isn't available
        on this site (matching :meth:`_search_keys`).
        """
        modern_url = f"{self.base_url}/rest/api/3/search/jql"
        payload: Dict[str, Any] = {
            "jql": jql,
            "fields": ["summary", "status"],
            "maxResults": max_results,
        }
        resp = self._post(modern_url, json=payload)
        if resp.status_code in (404, 410):
            log.warning(
                "Related-tickets: modern /search/jql unavailable "
                "(HTTP %d); falling back to legacy /search.",
                resp.status_code,
            )
            params = {
                "jql": jql,
                "fields": "summary,status",
                "maxResults": max_results,
            }
            resp = self._get(
                f"{self.base_url}/rest/api/3/search", params=params,
            )
        resp.raise_for_status()
        data = resp.json()
        out: List[RelatedIssue] = []
        for issue in data.get("issues", []):
            key = issue.get("key", "") or ""
            fields = issue.get("fields") or {}
            summary = fields.get("summary", "") or ""
            status = ((fields.get("status") or {}).get("name")) or ""
            out.append(RelatedIssue(
                key=key,
                summary=summary,
                url=f"{self.base_url}/browse/{key}" if key else "",
                status=status,
            ))
        return out

    def download_attachment_text(self, attachment: Attachment) -> Optional[str]:
        """Download an attachment and decode it as UTF-8 if plausible.

        Previously this method skipped any file whose declared size
        was over the limit. That dropped 11–18MB CheckIn logs entirely,
        and we miss real app metadata inside. Now we stream and stop
        once :data:`MAX_ATTACHMENT_BYTES` is reached — the start of a
        log is usually where ``packageName=…`` and version markers
        live.
        """
        if not _looks_text_like(attachment):
            log.debug("Skipping non-text attachment %s (%s)",
                      attachment.filename, attachment.mime_type)
            return None
        resp = self._get(attachment.content_url, stream=True)
        resp.raise_for_status()
        chunks: List[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_ATTACHMENT_BYTES:
                log.info(
                    "Truncating attachment %s at %d bytes "
                    "(declared size %d) — processing the head.",
                    attachment.filename,
                    MAX_ATTACHMENT_BYTES,
                    attachment.size,
                )
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return None

    def expand_zip_attachment(
        self, attachment: Attachment,
    ) -> List[Attachment]:
        """Download a zip and return one virtual :class:`Attachment`
        per text-like entry inside.

        Why: many BMW / Mercedes tickets attach their log bundles as
        zips (sherlog, X2ENext, BACKEND_LOGS …). The actual
        ``packageName=…`` or ``versionName=…`` lines we need to read
        live inside those zips. Without this step those signals are
        invisible to the extractor.

        Bounded by :data:`MAX_ZIP_BYTES` for the download and by
        :data:`MAX_ATTACHMENT_BYTES` per inner file. Returns an
        empty list when the zip is corrupt, too big, or unreadable —
        never raises.
        """
        if not attachment.filename.lower().endswith(".zip"):
            return []
        if attachment.size and attachment.size > MAX_ZIP_BYTES:
            log.info(
                "Skipping oversized zip %s (%d bytes > %d cap)",
                attachment.filename, attachment.size, MAX_ZIP_BYTES,
            )
            return []
        try:
            resp = self._get(attachment.content_url, stream=True)
            resp.raise_for_status()
        except Exception as exc:
            log.warning("Failed to download zip %s: %s",
                        attachment.filename, exc)
            return []

        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=128 * 1024):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > MAX_ZIP_BYTES:
                log.info(
                    "Zip %s exceeded %d-byte cap mid-download — skipping.",
                    attachment.filename, MAX_ZIP_BYTES,
                )
                return []

        import io
        import zipfile

        out: List[Attachment] = []
        try:
            with zipfile.ZipFile(io.BytesIO(bytes(buf))) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner_name = info.filename
                    lname = inner_name.lower()
                    if not any(lname.endswith(ext)
                               for ext in TEXT_LIKE_EXTENSIONS):
                        continue
                    if info.file_size > MAX_ATTACHMENT_BYTES:
                        log.info(
                            "Truncating %s inside zip %s (%d bytes)",
                            inner_name, attachment.filename,
                            info.file_size,
                        )
                    try:
                        with zf.open(info) as f:
                            data = f.read(MAX_ATTACHMENT_BYTES)
                    except Exception as exc:
                        log.warning(
                            "Bad zip entry %s in %s: %s",
                            inner_name, attachment.filename, exc,
                        )
                        continue
                    try:
                        text = data.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    out.append(Attachment(
                        filename=f"{attachment.filename}:{inner_name}",
                        mime_type=_guess_mime(inner_name),
                        size=info.file_size,
                        content_url=attachment.content_url,
                        text=text,
                    ))
        except zipfile.BadZipFile:
            log.warning("Bad zip file: %s", attachment.filename)
        except Exception as exc:
            log.warning("Failed to extract zip %s: %s",
                        attachment.filename, exc)
        log.info(
            "Expanded zip %s into %d text entry(ies).",
            attachment.filename, len(out),
        )
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _search_keys(self, jql: str, *, page_size: int) -> Iterable[str]:
        """Iterate issue keys for ``jql`` using the modern /search/jql.

        Atlassian deprecated the legacy ``/rest/api/3/search`` endpoint
        in favour of ``/rest/api/3/search/jql``, which uses
        ``nextPageToken`` instead of ``startAt``.
        """
        url = f"{self.base_url}/rest/api/3/search/jql"
        next_token: Optional[str] = None
        page = 0
        while True:
            page += 1
            payload: Dict[str, Any] = {
                "jql": jql,
                "fields": ["summary"],  # we hydrate fully via get_issue
                "maxResults": page_size,
            }
            if next_token:
                payload["nextPageToken"] = next_token
            resp = self._post(url, json=payload)
            if resp.status_code == 410 or resp.status_code == 404:
                # Endpoint not yet rolled out for this site → fall back.
                log.warning(
                    "Modern /search/jql unavailable (HTTP %d); "
                    "falling back to legacy /search.", resp.status_code,
                )
                yield from self._search_keys_legacy(jql, page_size=page_size)
                return
            resp.raise_for_status()
            data = resp.json()
            for issue in data.get("issues", []):
                key = issue.get("key")
                if key:
                    yield key
            next_token = data.get("nextPageToken")
            if not next_token or data.get("isLast"):
                return

    def _search_keys_legacy(self, jql: str, *, page_size: int) -> Iterable[str]:
        """Legacy /rest/api/3/search using startAt pagination."""
        url = f"{self.base_url}/rest/api/3/search"
        start_at = 0
        while True:
            params = {
                "jql": jql,
                "fields": "summary",
                "startAt": start_at,
                "maxResults": page_size,
            }
            resp = self._get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            issues = data.get("issues", [])
            for issue in issues:
                key = issue.get("key")
                if key:
                    yield key
            total = data.get("total", 0)
            start_at += len(issues)
            if start_at >= total or not issues:
                return

    def _build_issue(self, data: Dict[str, Any]) -> JiraIssue:
        fields = data.get("fields", {}) or {}
        assignee = fields.get("assignee") or {}
        comments_raw = (fields.get("comment") or {}).get("comments", [])
        attachments_raw = fields.get("attachment", []) or []

        comments: List[Comment] = []
        for c in comments_raw:
            comments.append(Comment(
                author=(c.get("author") or {}).get("displayName") or "?",
                body=adf_to_text(c.get("body")),
                created=c.get("created", ""),
            ))

        attachments: List[Attachment] = []
        for a in attachments_raw:
            mime = a.get("mimeType") or ""
            filename = a.get("filename") or "attachment"
            size = int(a.get("size") or 0)
            content_url = a.get("content") or ""
            attachments.append(Attachment(
                filename=filename,
                mime_type=mime or _guess_mime(filename),
                size=size,
                content_url=content_url,
            ))

        # Hydrate text content for plausible attachments, AND expand
        # zip attachments into their text-like entries so the regular
        # extractor sees ``packageName=...`` lines that are buried
        # inside log bundles (sherlog, BACKEND_LOGS, X2ENext zips, …).
        zip_expansions: List[Attachment] = []
        for att in attachments:
            if att.filename.lower().endswith(".zip"):
                try:
                    zip_expansions.extend(self.expand_zip_attachment(att))
                except Exception as exc:
                    log.warning(
                        "Failed to expand zip %s: %s",
                        att.filename, exc,
                    )
                continue  # the zip itself isn't text — only its contents
            try:
                att.text = self.download_attachment_text(att)
            except requests.RequestException as exc:
                log.warning("Failed to download attachment %s: %s",
                            att.filename, exc)
        # Tack the zip-extracted virtual attachments onto the issue so
        # ``_build_sources`` in the extractor sees them with full
        # attachment weight.
        attachments.extend(zip_expansions)

        issue_url = (
            f"{self.base_url}/browse/{data['key']}" if data.get("key") else ""
        )
        raw = dict(data)
        raw["_self_url"] = issue_url

        return JiraIssue(
            key=data.get("key", ""),
            summary=fields.get("summary", "") or "",
            description=adf_to_text(fields.get("description")),
            status=((fields.get("status") or {}).get("name")) or "",
            issue_type=((fields.get("issuetype") or {}).get("name")) or "",
            assignee_display_name=assignee.get("displayName"),
            assignee_email=assignee.get("emailAddress"),
            assignee_account_id=assignee.get("accountId"),
            created=fields.get("created", "") or "",
            updated=fields.get("updated", "") or "",
            comments=comments,
            attachments=attachments,
            raw=raw,
        )

    # ---- HTTP helpers (retry-wrapped) --------------------------------

    @retry(logger=log)
    def _get(self, url: str, *, params: Optional[Dict[str, Any]] = None,
             stream: bool = False) -> requests.Response:
        return self._session.get(
            url, params=params, timeout=self.timeout, stream=stream,
        )

    @retry(logger=log)
    def _post(self, url: str, *, json: Optional[Dict[str, Any]] = None,
              ) -> requests.Response:
        return self._session.post(url, json=json, timeout=self.timeout)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _guess_mime(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"


def _looks_text_like(att: Attachment) -> bool:
    name = (att.filename or "").lower()
    for ext in TEXT_LIKE_EXTENSIONS:
        if name.endswith(ext):
            return True
    mime = (att.mime_type or "").lower()
    return any(mime.startswith(p) for p in TEXT_LIKE_MIME_PREFIXES)
