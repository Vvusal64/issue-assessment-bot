"""Application metadata extraction.

Two backends:

* ``RegexExtractor`` — deterministic, source-aware heuristics. Patterns
  are scored by where they were found (an explicit ``versionName=`` in a
  log file is high; a stray dotted-package-looking string in a comment
  is low). Returns the best candidate per field plus a confidence
  level for the whole extraction.

* ``ClaudeExtractor`` — optional LLM fallback. Used only when the regex
  pass returns ``low`` confidence or any field is ``None``. Builds a
  compact prompt from the same source bundle and asks Claude to return
  strict JSON. Refuses to invent values: missing fields stay ``null``.

The orchestrator instantiates :class:`HybridExtractor`, which composes
both backends behind a uniform interface.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .app_name_resolver import AppNameResolver, default_resolver
from .catalog import KnownAppCatalog
from .jira_client import JiraIssue, StructuredFields
from .utils import truncate


log = logging.getLogger("bot.extractor")


CONFIDENCE_LEVELS = ("low", "medium", "high")


@dataclass
class Extraction:
    app_name: Optional[str] = None
    app_version: Optional[str] = None
    app_versions: List[str] = field(default_factory=list)
    package_name: Optional[str] = None
    version_code: Optional[str] = None
    environment_name: Optional[str] = None
    confidence: str = "low"
    evidence: Dict[str, str] = field(default_factory=dict)
    backend: str = "regex"
    # Outcome of the Admin Portal cross-check. The bot publishes
    # "high-confidence" results ONLY when this is "verified" — i.e.
    # the package coordinate exists in the portal AND the extracted
    # version is one of the portal's published versions. Other
    # values:
    #   * ``"version_mismatch"`` — package is in the portal but the
    #     extracted version isn't on the approved list. The name
    #     still gets overridden to the portal's value.
    #   * ``"no_match"`` — the package is not registered in the portal
    #     (or no package was extracted). Triggers a confidence
    #     downgrade and a "manual validation required" Slack note.
    #   * ``"not_checked"`` — no portal resolver was configured /
    #     reachable. Legacy behaviour preserved.
    admin_portal_status: str = "not_checked"
    admin_portal_versions: List[str] = field(default_factory=list)

    def missing_fields(self) -> List[str]:
        return [
            name for name, value in (
                ("app_name", self.app_name),
                ("app_version", self.app_version),
                ("package_name", self.package_name),
            )
            if not value
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_versions": list(self.app_versions),
            "package_name": self.package_name,
            "version_code": self.version_code,
            "environment_name": self.environment_name,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "backend": self.backend,
            "admin_portal_status": self.admin_portal_status,
            "admin_portal_versions": list(self.admin_portal_versions),
        }


class ExtractorBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def extract(self, issue: JiraIssue) -> Extraction:
        ...


@dataclass
class _Source:
    label: str
    text: str
    weight: int


def _build_sources(issue: JiraIssue) -> List[_Source]:
    sources: List[_Source] = []

    for att in issue.attachments:
        if att.text:
            sources.append(_Source(
                label=f"attachment:{att.filename}",
                text=att.text,
                weight=3,
            ))

    if issue.description:
        sources.append(_Source(
            label="description",
            text=issue.description,
            weight=2,
        ))

    for c in issue.comments:
        sources.append(_Source(
            label=f"comment:{c.author}",
            text=c.body,
            weight=2,
        ))

    if issue.summary:
        sources.append(_Source(
            label="summary",
            text=issue.summary,
            weight=1,
        ))

    return sources


_PACKAGE_RE = re.compile(
    r"\b(?:[a-z][a-z0-9_]{1,})(?:\.[a-zA-Z][a-zA-Z0-9_]{0,})+\b"
)


def _is_in_url_or_email_context(text: str, position: int) -> bool:
    i = position
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] in "._-"):
        i -= 1

    if i >= 3 and text[i - 3:i] == "://":
        return True

    if i >= 1 and text[i - 1] == "@":
        return True

    return False


_PACKAGE_LABELED_RE = re.compile(
    r"(?ix)"
    r"(?:package(?:\s*name)?|pkg|applicationId|app[_\s-]?id|process)"
    r"\s*[:=]\s*\[?"
    r"([a-z][a-z0-9_]+(?:\.[a-zA-Z0-9_]+){1,})"
    r"\]?"
)


_VERSION_BARE_RE = re.compile(
    r"\b(?:v|version[:\s]*)?"
    r"(\d{1,4}(?:\.\d{1,4}){1,3}(?:[-_+.][A-Za-z0-9]+)?)\b"
)


_VERSION_LABELED_RE = re.compile(
    r"(?ix)"
    r"(?<![\w-])"
    r"(?:versionName|application[_\s-]?version|app[_\s-]?version)"
    r"\s*[:=]\s*['\"]?v?(\d{1,4}(?:\.\d{1,4}){0,3}"
    r"(?:[-_+.][A-Za-z0-9]+)?)"
)


_VERSION_CODE_LABELED_RE = re.compile(
    r"(?ix)"
    r"(?<![\w-])"
    r"(?:versionCode|version_code|app[_\s-]?version[_\s-]?code|"
    r"application[_\s-]?version[_\s-]?code)"
    r"\s*[:=]\s*['\"]?(\d{1,12})"
)


_ENVIRONMENT_LABELED_RE = re.compile(
    r"(?ix)"
    r"(?:environment(?:\s*name)?|test\s*environment|env)"
    r"\s*[:=]\s*['\"]?([A-Za-z0-9_.\- /()]{2,80})"
)


_VERSION_FROM_PACKAGE_LINE_RE = re.compile(
    r"(?ix)"
    r"package\s*[:=]\s*\[?"
    r"[a-z][a-z0-9_]+(?:\.[a-zA-Z0-9_]+){1,}"
    r"\]?"
    r"(?:\s*\([^)]*\))?"
    r"\s+v\d+\s*"
    r"\((\d+(?:\.\d+){1,3}(?:[-_+.][A-Za-z0-9]+)?)\)"
)


_VERSION_FROM_VERSION_LINE_RE = re.compile(
    r"(?ix)"
    r"(?<![\w-])version\s*[:=]\s*v\d+\s*"
    r"\((\d+(?:\.\d+){1,3}(?:[-_+.][A-Za-z0-9]+)?)\)"
)


_CRASH_ID_VERSION_RE = re.compile(
    r"(?i)\|\s*ver\s*=\s*v?\d*\s*"
    r"\((\d+(?:\.\d+){1,3}(?:[-_+.][A-Za-z0-9]+)?)\)"
)


# Quoted app name following a context word like "game", "app",
# "application". Catches summaries such as
#     game "unblock it" was played
#     the app "Spotify" crashed
# Handles odd closing-quote characters seen in real BMW tickets
# (´, », curly quotes, …).
_QUOTED_APP_NAME_RE = re.compile(
    r"(?ix)"
    r"\b(?:game|app(?:lication)?|application)\b"  # context cue
    r"\s+"
    r"[\"'`«‘“]"                         # opening quote
    r"(?P<name>[^\"'`«»‘’“”\n]{2,60}?)"
    r"[\"'`»’”´]"                        # closing quote
)

# Mercedes / SyncTool descriptions often phrase the target app as:
#     Go to the app of "Genie".
# The older quoted-name pattern only matched ``the app "Genie"``.
# This explicit variant is required so the bot can anchor on the
# app mentioned in the reproduction steps before asking the Admin
# Portal to resolve the canonical package/version.
_APP_OF_QUOTED_NAME_RE = re.compile(
    r"(?ix)"
    r"\b(?:app(?:lication)?|game)\s+of\s+"
    r"[\"'`«‘“]"
    r"(?P<name>[^\"'`«»‘’“”\n]{2,60}?)"
    r"[\"'`»’”´]"
)


_NL_VERSION_PAIR_RE = re.compile(
    r"(?ix)"
    r"(?:^|[\s.,;:!?(/\\])"
    r"((?:[A-Z][\w&\-]{0,30})(?:\s+[A-Z0-9][\w&\-]{0,30}){0,3})"
    r"\s+(?:latest\s+(?:available\s+)?version|current\s+version)"
    r"\s+is\s+(?:already\s+)?installed[\s.:,\-]*"
    r"(\d+(?:\.\d+){1,3}(?:[-_+.][A-Za-z0-9]+)?)"
)

# BMW / Forvia reporter template: a description line like
#   "Apps (if specific version, please mention): Tagesschau Automotive v1.0.6"
# or just
#   "Apps (if specific version, please mention): Nextory"
# captures the canonical app name (+ version when present) from a
# structured description field, which is one of the strongest signals
# we have outside Jira's actual custom fields.
_APPS_TEMPLATE_RE = re.compile(
    r"(?im)^\s*Apps?\s*"
    r"(?:\([^)]*\))?\s*"           # optional parenthetical hint
    r"[:=]\s*"
    r"(?P<name>[A-Z][^\n]{1,80}?)"
    r"(?:\s+v\s*(?P<ver>\d+(?:\.\d+){1,3}(?:[-_+.][A-Za-z0-9]+)?))?"
    r"\s*$"
)

# BMW STAT_APPS_TEXT inventory entry, as found inside CheckIn.txt
# attachments on BMW tickets:
#     com.forvia.zoomapp.push_notifications;1.0.8.9RC11;270426 06:09
# (slashes between entries, semicolons or colons between fields, six-
# digit YYMMDD timestamp). Each match pairs a package with its EXACT
# published version — much stronger than seeing either alone, because
# they came from the same field of the same line.
_BMW_INVENTORY_RE = re.compile(
    r"(?x)"
    r"(?P<package>"
    r"  [a-z][a-z0-9_]+"
    r"  (?:\.[a-zA-Z0-9_]+){1,}"
    r")"
    r"\s*[;:]\s*"
    r"(?P<version>"
    r"  \d+(?:\.\d+){0,3}"
    r"  (?:[-_+.]?[A-Za-z0-9]+)?"  # suffix attached or separated
    r")"
    r"\s*[;:]\s*"
    r"\d{6}"  # YYMMDD timestamp
)


_APP_NAME_LABELED_RE = re.compile(
    r"(?im)^\s*(?:app(?:lication)?(?:[\s_-]?name)?|appLabel|label)"
    r"\s*[:=]\s*['\"]?([^\n'\"]{2,80}?)['\"]?\s*$"
)


_PACKAGE_PHRASE_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"  crash(?:ed|ing)?\s+(?:due\s+to|caused\s+by|by|in|of)"
    r"  | caused\s+by"
    r"  | due\s+to"
    r"  | in\s+package"
    r"  | app\s+package"
    r"  | package(?:\s+name)?\s+is"
    r")"
    r"\s+\[?\s*"
    r"([a-z][a-z0-9_]+(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)"
)


_ENV_TAG_RE = re.compile(
    r"(?i)^\s*("
    r"idc\w*"
    r"|cde(?:[-_ ]?\w+)?"
    r"|mgu\w*|hu[-_ ]?mgu\w*|mgua"
    r"|bmw(?:\s+\w+)*"
    r"|vf\d+"
    r"|v\d+(?:\.\d+)*"
    r"|vn|us|eu|emea|asia|apac|na|sa"
    r"|prod(?:uction)?|stag(?:e|ing)|qa|test|dev(?:elop(?:ment)?)?"
    r"|android|ios|linux|windows"
    r")\s*$"
)


_VERSION_BLOCKLIST = {"1.0", "0.0", "0.0.0"}


_PACKAGE_BLOCKLIST_PREFIXES = (
    "android.", "androidx.", "java.", "javax.", "kotlin.",
    "kotlinx.", "com.android.", "com.google.android.gms.",
    "com.google.firebase.", "dalvik.", "sun.",
    # BMW / Forvia / Aptoide platform & system services. These show
    # up in the logs of nearly every BMW ticket because they're the
    # in-vehicle platform interacting with the actual third-party
    # app, but they are NEVER the app under test. BMW-3460 picked
    # ``com.bmwgroup.idnext.bmwcarplayinterface.service`` and called
    # the app "Bmwgroup"; the bug was actually about the game
    # ``com.marketjs.unblockit``.
    "com.bmwgroup.", "com.bmw.",
    "com.aptoide.", "com.appning.",
    "com.faurecia.",
    # Aptoide platform integrations on the head unit.
    "cm.aptoide.",
)


def _candidate_score(source: _Source, *, signal: str) -> int:
    base = source.weight
    bonus = {"phrase": 5, "labeled": 3, "bare": 0}[signal]
    return base + bonus


@dataclass
class _Candidate:
    value: str
    score: int
    source_label: str


def _best(candidates: List[_Candidate]) -> Optional[_Candidate]:
    if not candidates:
        return None
    # On a score tie, prefer the LONGER value — for packages it's the
    # more-specific sub-variant (``com.forvia.zoomapp.rsedemo`` beats
    # ``com.forvia.zoomapp``); for app names, a longer alias is more
    # specific too. The score is still the primary ordering.
    return max(candidates, key=lambda c: (c.score, len(c.value)))


def _scan_packages(sources: List[_Source]) -> List[_Candidate]:
    out: List[_Candidate] = []

    for src in sources:
        for m in _PACKAGE_PHRASE_RE.finditer(src.text):
            pkg = m.group(1).strip()
            if _is_acceptable_package(pkg):
                out.append(_Candidate(
                    pkg,
                    _candidate_score(src, signal="phrase"),
                    src.label,
                ))

        for m in _PACKAGE_LABELED_RE.finditer(src.text):
            pkg = m.group(1).strip()
            if _is_acceptable_package(pkg):
                out.append(_Candidate(
                    pkg,
                    _candidate_score(src, signal="labeled"),
                    src.label,
                ))

        if src.weight >= 2 and not _is_structured_data_attachment(src.label):
            # Bare scan only on free-text sources. Structured-data
            # attachments (.json / .xml / .yaml / …) are full of
            # dotted keys that look like packages but aren't — let
            # labeled and phrase patterns above carry those.
            for m in _PACKAGE_RE.finditer(src.text):
                if _is_in_url_or_email_context(src.text, m.start()):
                    continue

                pkg = m.group(0).strip()

                if _bare_acceptable_package(pkg):
                    out.append(_Candidate(
                        pkg,
                        _candidate_score(src, signal="bare"),
                        src.label,
                    ))

    return out


def _scan_versions(sources: List[_Source]) -> List[_Candidate]:
    out: List[_Candidate] = []

    for src in sources:
        for pattern in (
            _VERSION_FROM_PACKAGE_LINE_RE,
            _VERSION_FROM_VERSION_LINE_RE,
            _CRASH_ID_VERSION_RE,
        ):
            for m in pattern.finditer(src.text):
                v = m.group(1).strip()
                if v in _VERSION_BLOCKLIST:
                    continue

                out.append(_Candidate(
                    v,
                    _candidate_score(src, signal="phrase"),
                    src.label,
                ))

        for m in _VERSION_LABELED_RE.finditer(src.text):
            v = m.group(1).strip()
            if v in _VERSION_BLOCKLIST:
                continue

            out.append(_Candidate(
                v,
                _candidate_score(src, signal="labeled"),
                src.label,
            ))

    return out


def _scan_version_codes(sources: List[_Source]) -> List[_Candidate]:
    out: List[_Candidate] = []
    for src in sources:
        for m in _VERSION_CODE_LABELED_RE.finditer(src.text):
            code = m.group(1).strip()
            if code:
                out.append(_Candidate(
                    code,
                    _candidate_score(src, signal="labeled"),
                    src.label,
                ))
    return out


def _scan_environments(sources: List[_Source]) -> List[_Candidate]:
    out: List[_Candidate] = []
    for src in sources:
        for m in _ENVIRONMENT_LABELED_RE.finditer(src.text):
            env = m.group(1).strip().strip('"\'')
            if env:
                out.append(_Candidate(
                    env,
                    _candidate_score(src, signal="labeled"),
                    src.label,
                ))
    return out


def _scan_app_names(sources: List[_Source]) -> List[_Candidate]:
    out: List[_Candidate] = []

    for src in sources:
        for m in _APP_NAME_LABELED_RE.finditer(src.text):
            name = m.group(1).strip()

            if not _is_acceptable_app_name(name):
                continue

            out.append(_Candidate(
                name,
                _candidate_score(src, signal="labeled"),
                src.label,
            ))

    return out


def _scan_quoted_app_names(
    sources: List[_Source],
) -> List[Tuple[str, _Source]]:
    """Find quoted app names following a context cue like
    ``game``, ``app``, or ``application``.

    BMW-3460 had a summary like ``game "unblock it" was played``
    — the bot has no chance of recognising "Unblock It" via the
    catalog (it's a third-party game we never indexed) unless we
    pull it out of the quotes. Returns ``(name, source)`` pairs,
    Title-Cased for display ("unblock it" → "Unblock It").
    """
    out: List[Tuple[str, _Source]] = []

    def _display_name(raw: str) -> Optional[str]:
        raw = raw.strip()
        if not raw or len(raw) < 2:
            return None
        # Title-case for display; preserve non-alpha tokens as-is.
        tokens = [t for t in raw.split() if t]
        display = " ".join(
            (t[0].upper() + t[1:].lower()) if t.isalpha() else t
            for t in tokens
        )
        if not _is_acceptable_app_name(display):
            return None
        return display

    for src in sources:
        for pattern in (_QUOTED_APP_NAME_RE, _APP_OF_QUOTED_NAME_RE):
            for m in pattern.finditer(src.text):
                display = _display_name(m.group("name"))
                if display:
                    out.append((display, src))
    return out


def _scan_bmw_inventory(
    sources: List[_Source],
) -> List[Tuple[str, str, _Source]]:
    """Find BMW STAT_APPS_TEXT inventory entries.

    These appear inside ``CheckIn.txt`` attachments on BMW tickets
    and reliably pair a package with its installed version on the
    same line. Returns ``(package, version, source)`` triples — the
    package and version came from the SAME entry, so the caller can
    tag them both with the package and let the summary anchor keep
    them or drop them as a pair.
    """
    out: List[Tuple[str, str, _Source]] = []
    for src in sources:
        for m in _BMW_INVENTORY_RE.finditer(src.text):
            pkg = m.group("package")
            ver = m.group("version")
            if not _is_acceptable_package(pkg):
                continue
            if ver in _VERSION_BLOCKLIST:
                continue
            out.append((pkg, ver, src))
    return out


def _scan_apps_template(
    sources: List[_Source],
) -> List[Tuple[str, Optional[str], _Source]]:
    """Find BMW-reporter "Apps: <name> v<version>" lines.

    Returns ``(name, version_or_None, source)`` triples. Both
    summary and description are searched; descriptions are by far
    the most common location.
    """
    out: List[Tuple[str, Optional[str], _Source]] = []
    for src in sources:
        for m in _APPS_TEMPLATE_RE.finditer(src.text):
            name = (m.group("name") or "").strip()
            version = (m.group("ver") or "").strip() or None
            if not _is_acceptable_app_name(name):
                continue
            if version and version in _VERSION_BLOCKLIST:
                version = None
            out.append((name, version, src))
    return out


def _scan_natural_language_pairs(
    sources: List[_Source],
) -> List[Tuple[str, str, _Source]]:
    out: List[Tuple[str, str, _Source]] = []

    for src in sources:
        for m in _NL_VERSION_PAIR_RE.finditer(src.text):
            name = m.group(1).strip()
            version = m.group(2).strip()

            if not _is_acceptable_app_name(name):
                continue

            if version in _VERSION_BLOCKLIST:
                continue

            out.append((name, version, src))

    return out


def _is_acceptable_app_name(name: str) -> bool:
    if not (1 < len(name) < 80):
        return False

    if _ENV_TAG_RE.match(name):
        return False

    if _PACKAGE_RE.fullmatch(name):
        return False

    return True


_DOMAIN_TLDS = {
    "com", "org", "net", "io", "co", "uk", "de", "fr", "ru", "cn",
    "jp", "kr", "in", "br", "ca", "us", "edu", "gov", "info", "biz",
}


def _is_acceptable_package(pkg: str) -> bool:
    if not pkg or "." not in pkg:
        return False

    if any(pkg.startswith(p) for p in _PACKAGE_BLOCKLIST_PREFIXES):
        return False

    if re.fullmatch(r"\d+(?:\.\d+)+", pkg):
        return False

    if any(c in pkg for c in ("/", "\\", "@", ":", " ")):
        return False

    if pkg.lower().startswith(("http", "ftp", "file")):
        return False

    parts = pkg.split(".")

    if len(parts) < 2:
        return False

    if len(parts) == 2 and parts[1].lower() in _DOMAIN_TLDS:
        return False

    return True


def _bare_acceptable_package(pkg: str) -> bool:
    """Stricter validation for bare regex hits (no labeled/phrase
    context). Requires at least 3 segments, which kills false
    positives like ``term.show_file`` plucked out of a JSON config
    that happens to contain a dotted key. Two-segment packages
    (``eimycarolaym.blanco``, ``tunein.player``) are still accepted
    when they come from labeled or phrase patterns or from the
    catalog — just not from bare scans.
    """
    if not _is_acceptable_package(pkg):
        return False
    return pkg.count(".") >= 2


# Attachment filename suffixes that are structured data, not
# free-form text. We deliberately skip *bare* package scans inside
# these because their dotted keys ("term.show_file", "data.input.x")
# look package-shaped but never are. Labeled/phrase patterns
# (``Process: …``, ``packageName=…``) still fire there.
_STRUCTURED_ATTACHMENT_SUFFIXES = (
    ".json", ".xml", ".yaml", ".yml", ".csv", ".html", ".htm",
    ".manifest", ".plist",
)


def _is_structured_data_attachment(label: str) -> bool:
    if not label.startswith("attachment:"):
        return False
    name = label[len("attachment:"):].strip().lower()
    return any(name.endswith(s) for s in _STRUCTURED_ATTACHMENT_SUFFIXES)


# Reverse-DNS prefixes (top-level domain, country code, common tech
# prefixes) that get stripped from the START of a package coordinate
# before deriving a human-readable app name.
_PACKAGE_PREFIX_NOISE = {
    "com", "org", "net", "io", "co", "us", "de", "uk", "fr",
    "google", "amazon", "android",
}
# Platform variant suffixes that get stripped from the END.
_PACKAGE_SUFFIX_NOISE = {
    "android", "app", "auto", "automotive", "cars", "car", "mobile",
    "client", "lite", "pro", "free", "premium", "edition",
    "push_notifications", "push",
}


def _derive_name_from_package(package: Optional[str]) -> Optional[str]:
    """Turn a package coordinate into a human-readable app name.

    Offline approximation of an Admin Portal lookup. The result is
    intentionally MARKED as "derived" by the caller so confidence
    can be capped — derive is a best-effort heuristic, not a
    verified mapping.

    Strategy:
      1. Split on ``.`` and strip reverse-DNS prefix noise (``com``,
         ``de``, ``us`` …) from the FRONT.
      2. Strip platform-variant noise (``auto``, ``automotive``,
         ``android``, ``app``, ``mobile``, ``push_notifications`` …)
         from the BACK.
      3. **Also drop mid-segment noise** (``android``, ``automotive``
         that appear sandwiched in the middle, e.g.
         ``de.spiegel.android.automotive.mmo``). This is the BMW-3438
         fix — without it the longest middle segment "automotive"
         got picked.
      4. Repetition signal (``radioline.android.radioline.auto`` →
         Radioline).
      5. Otherwise prefer the LEFTMOST segment of length ≥ 4 —
         brand-first, since real Android packages are reverse-DNS
         ordered with the company/product name near the front.
      6. If every remaining segment is short (≤3 chars), fall back
         to the longest one of length ≥ 4. If none ≥ 4, ``None``.

    Examples:
        ``de.spiegel.android.automotive.mmo`` → ``"Spiegel"``
        ``com.spotify.music``                 → ``"Spotify"``
        ``com.gtl.nextory``                   → ``"Nextory"`` (gtl is <4)
        ``com.radioline.android.radioline.auto`` → ``"Radioline"``
        ``de.tagesschau``                     → ``"Tagesschau"``
        ``de.tagesschau.automotive``          → ``"Tagesschau"``
    """
    if not package or "." not in package:
        return None
    segments = [s for s in package.split(".") if s]
    if len(segments) < 2:
        return None

    # 1. Strip prefix noise.
    stripped = list(segments)
    while stripped and stripped[0].lower() in _PACKAGE_PREFIX_NOISE:
        stripped.pop(0)
    # 2. Strip suffix noise.
    while stripped and stripped[-1].lower() in _PACKAGE_SUFFIX_NOISE:
        stripped.pop()
    # 3. Drop mid-segment noise (anywhere in the middle).
    stripped = [
        s for s in stripped
        if s.lower() not in _PACKAGE_SUFFIX_NOISE
        and s.lower() not in _PACKAGE_PREFIX_NOISE
    ]

    candidate: Optional[str] = None
    if stripped:
        # 4. Repetition signal first.
        counts: Dict[str, int] = {}
        for s in stripped:
            counts[s.lower()] = counts.get(s.lower(), 0) + 1
        if max(counts.values()) > 1:
            repeated = [s for s in stripped if counts[s.lower()] > 1]
            candidate = max(repeated, key=len)

        # 5. Brand-first: leftmost segment of length ≥ 4.
        if not candidate:
            for s in stripped:
                if len(s) >= 4:
                    candidate = s
                    break

        # 6. Fallback: any segment of length ≥ 4 (would be rightmost
        # now since loop above stopped at the first leftmost match).
        if not candidate:
            longer = [s for s in stripped if len(s) >= 4]
            if longer:
                candidate = max(longer, key=len)

    if not candidate or len(candidate) < 4:
        return None
    return candidate[0].upper() + candidate[1:].lower()


def _catalog_app_from_label(source_label: str) -> Optional[str]:
    """Pull the embedded app name out of any tagged source label.

    Handles both ``catalog[App]:src`` (from the KnownAppCatalog scan)
    and ``apps_template[App]:src`` (from the BMW reporter-template
    scan). Returns ``None`` for non-tagged sources (description,
    summary, attachment:filename, labeled hits, …).
    """
    for prefix in ("catalog[", "apps_template["):
        if source_label.startswith(prefix):
            end = source_label.find("]", len(prefix))
            if end > 0:
                return source_label[len(prefix):end]
    return None


def _bmw_inventory_pkg_from_label(source_label: str) -> Optional[str]:
    """Pull the embedded package coordinate out of an inventory label.

    Supports both regular CheckIn labels and issue-anchor boosted labels:
    ``bmw_inventory[pkg]:src`` and ``issue_anchor_inventory[pkg]:src``.
    """
    for prefix in ("bmw_inventory[", "issue_anchor_inventory["):
        if not source_label.startswith(prefix):
            continue
        end = source_label.find("]", len(prefix))
        if end <= 0:
            return None
        return source_label[len(prefix):end]
    return None



# ----------------------------------------------------------------------
# Generic issue-text anchors
# ----------------------------------------------------------------------
#
# Not every real app is already in ``known_apps.json``. BMW tickets often
# name the app clearly in the title/description, while the CheckIn.txt
# attachment contains a system-wide inventory of dozens of apps. If the
# app named in the ticket is not in the catalog (example: Bloomberg), the
# old logic had no summary anchor and could incorrectly pick another app
# from the same inventory (example: Der Spiegel). These helpers create a
# lightweight anchor directly from issue text and use it to select the
# matching CheckIn inventory entry.
_SUMMARY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&+.-]{2,50}")

_APP_TOKEN_STOPWORDS = {
    "app", "apps", "application", "applications", "bug", "issue", "error",
    "occurred", "displayed", "after", "play", "playing", "news", "now",
    "open", "launch", "start", "manager", "terms", "term", "use", "missing",
    "description", "including", "legal", "notice", "information", "store",
    "connected", "drive", "connecteddrive", "vehicle", "online", "unknown",
    "platform", "observed", "expected", "behaviour", "behavior", "result",
    "preconditions", "actions", "please", "attached", "video", "picture",
    "multiple", "lifecycle", "lifecycles", "downloaded", "from", "press",
    "pressed", "another", "one", "them", "can", "without", "message",
}

_ACTION_APP_RE = re.compile(
    r"(?im)\b(?:open|launch|start|go\s+to|select|click|enter)\s+"
    r"(?:the\s+)?(?:app\s+of\s+)?[\"'`«‘“]?"
    r"(?P<name>[A-Za-z][A-Za-z0-9&+.\- ]{2,50}?)"
    r"(?:\s+app)?[\"'`»’”´]?"
    r"(?:\s|$|[.,;:])"
)

_DOWNLOADED_APP_RE = re.compile(
    r"(?im)\b(?P<name>[A-Za-z][A-Za-z0-9&+.\- ]{2,50}?)\s+"
    r"(?:app|application)\s+was\s+downloaded\b"
)


def _normalise_app_token(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def _clean_app_phrase(raw: str) -> Optional[str]:
    """Return a concise app-name phrase from free text.

    The phrase is intentionally conservative: it strips common ticket
    words and keeps the first meaningful brand-like token/phrase. This
    turns "Bloomberg APP_An error..." and "open Bloomber app" into
    "Bloomberg", while avoiding generic tokens like "Error" or "App".
    """
    if not raw:
        return None
    raw = raw.replace("_", " ").replace("-", " ")
    tokens = []
    for m in _SUMMARY_TOKEN_RE.finditer(raw):
        tok = m.group(0).strip(" .,:;()[]{}\"'`")
        norm = _normalise_app_token(tok)
        if not norm or norm in _APP_TOKEN_STOPWORDS:
            continue
        if norm.isdigit() or len(norm) < 3:
            continue
        tokens.append(tok)
    if not tokens:
        return None

    # Keep up to two words for app names like "Car Eats", but one word
    # is enough for the package-token match used by CheckIn inventory.
    phrase = " ".join(tokens[:2])
    return phrase.strip() or None


def _issue_app_anchor_names(sources: List[_Source]) -> List[str]:
    """Extract app-name anchors from summary/description text.

    This is NOT the final metadata source. It is only an anchor used to
    decide which CheckIn.txt inventory row belongs to the ticket. The
    final package/version still comes from the inventory row and/or
    Admin Portal verification.
    """
    out: List[str] = []

    def _add(raw: Optional[str]) -> None:
        cleaned = _clean_app_phrase(raw or "")
        if not cleaned:
            return
        if cleaned.lower() not in {x.lower() for x in out}:
            out.append(cleaned)

    for src in sources:
        if src.label == "summary":
            summary = src.text or ""
            # Strong BMW title format:
            #   Bloomberg APP_An error occurred...
            #   Bloomberg_App_...
            m = re.match(
                r"^\s*(?P<name>[A-Za-z][A-Za-z0-9&+.\- ]{2,50}?)"
                r"\s*[_\- ]+APP(?:[_\- ]|$)",
                summary,
                flags=re.IGNORECASE,
            )
            if m:
                _add(m.group("name"))
            else:
                # Fallback: the first meaningful token in the title.
                _add(summary.split("_", 1)[0])

        if src.label in {"summary", "description"}:
            for pat in (_ACTION_APP_RE, _DOWNLOADED_APP_RE):
                for m in pat.finditer(src.text or ""):
                    _add(m.group("name"))

    return out


def _issue_app_anchor_tokens(sources: List[_Source]) -> set:
    tokens = set()
    for name in _issue_app_anchor_names(sources):
        for m in _SUMMARY_TOKEN_RE.finditer(name):
            norm = _normalise_app_token(m.group(0))
            if norm and norm not in _APP_TOKEN_STOPWORDS and len(norm) >= 3:
                tokens.add(norm)
    return tokens


def _package_matches_anchor_tokens(package: str, tokens: set) -> bool:
    """Return True when a package clearly belongs to an app anchor.

    Example:
      anchor token "bloomberg" matches "bloomberg.android.plus".
      anchor token "genie" matches "com.ktmusic.geniemusic" only if the
      segment contains that token.
    """
    if not package or not tokens:
        return False
    parts = [
        _normalise_app_token(p)
        for p in re.split(r"[._\-]+", package)
        if _normalise_app_token(p)
    ]
    joined = _normalise_app_token(package)
    for token in tokens:
        if token in parts:
            return True
        if any(part.startswith(token) or token.startswith(part) for part in parts if len(part) >= 4):
            return True
        if len(token) >= 5 and token in joined:
            return True
    return False


def _display_name_from_anchor_tokens(tokens: set, package: Optional[str] = None) -> Optional[str]:
    if not tokens:
        return None
    if package:
        pkg_norm = _normalise_app_token(package)
        matching = [t for t in tokens if t and t in pkg_norm]
        if matching:
            token = max(matching, key=len)
            return token[:1].upper() + token[1:]
    token = max(tokens, key=len)
    return token[:1].upper() + token[1:]


class RegexExtractor(ExtractorBackend):
    """Heuristic extractor — fast, deterministic, transparent."""

    name = "regex"
    _STRONG_SCORE = 5

    def __init__(
        self, *,
        app_name_resolver: Optional[AppNameResolver] = None,
        catalog: Optional[KnownAppCatalog] = None,
    ) -> None:
        self._app_name_resolver: AppNameResolver = (
            app_name_resolver or default_resolver
        )
        # Lazy default — load the bundled catalog only if no override.
        self._catalog: KnownAppCatalog = (
            catalog if catalog is not None else KnownAppCatalog.load_default()
        )

    # ------------------------------------------------------------------
    # Consistency helpers
    # ------------------------------------------------------------------

    def _name_consistent_with_package(
        self, name: Optional[str], package: Optional[str],
    ) -> bool:
        """Return True if ``name`` is consistent with ``package``.

        A catalog name is "consistent" with a package when EITHER:

          * the catalog entry for the name lists that package as one
            of its variants (exact match), OR
          * one of the catalog's registered packages is a strict
            dotted prefix of the queried package — this catches
            sub-variants like ``com.forvia.zoomapp.rsedemo`` against
            the catalog's parent ``com.forvia.zoomapp``. BMW-3455
            relied on this: the ticket's labeled
            ``packageName=com.forvia.zoomapp.rsedemo`` is the exact
            published artefact for the same "Zoom" family.

        Names that aren't in the catalog at all are considered
        consistent — we have no constraint to apply. This is what
        stops a stray "YouTube" mention overriding the Radioline
        package on BMW-3401.
        """
        if not name:
            return False
        if not package:
            return True
        cat_app = self._catalog.find_by_name(name)
        if cat_app is None:
            return True
        catalog_packages = {
            v.package.lower() for v in cat_app.packages if v.package
        }
        if not catalog_packages:
            return True
        pkg_lower = package.strip().lower()
        if pkg_lower in catalog_packages:
            return True
        return any(
            pkg_lower.startswith(cat_pkg + ".")
            for cat_pkg in catalog_packages
        )

    # ------------------------------------------------------------------
    # Summary-app anchor — keep extraction tied to the headline.
    # ------------------------------------------------------------------

    def _summary_anchor_packages(self, sources: List[_Source]) -> set:
        """Return the union of package coordinates from every catalog
        entry whose name appears in the summary OR shares a family
        with a name that does.

        Example: a summary that says "Tagesschau Rendering is too
        small" maps to ``{de.tagesschau, de.tagesschau.automotive}``
        — Tagesschau's own packages PLUS Tagesschau Automotive's,
        because the two entries share the same head word.
        """
        summary_text = "\n".join(
            s.text for s in sources if s.label == "summary"
        )
        if not summary_text:
            return set()
        seen_apps = set()
        for cmatch in self._catalog.find_in_text(summary_text):
            seen_apps.add(cmatch.app.name)
        if not seen_apps:
            return set()
        # Expand to family-related catalog entries: any entry whose
        # name shares the same first token as one of ``seen_apps``.
        family_names: set = set(seen_apps)
        for seen in list(seen_apps):
            head = seen.split()[0].lower() if seen.split() else ""
            if not head or len(head) < 3:
                continue
            for cat_app in self._catalog.apps:
                first_token = (
                    cat_app.name.split()[0].lower()
                    if cat_app.name.split() else ""
                )
                if first_token == head:
                    family_names.add(cat_app.name)
        anchor: set = set()
        for app_name in family_names:
            cat_app = self._catalog.find_by_name(app_name)
            if not cat_app:
                continue
            for v in cat_app.packages:
                if v.package:
                    anchor.add(v.package.lower())
        return anchor

    @staticmethod
    def _in_anchor(pkg_value: str, anchor: set) -> bool:
        v = pkg_value.lower()
        if v in anchor:
            return True
        return any(v.startswith(a + ".") for a in anchor)

    def _anchor_packages(
        self, candidates: List[_Candidate], anchor: set,
    ) -> List[_Candidate]:
        """Filter package candidates by the summary anchor.

        Strict mode: if no package candidate is in the summary
        family, return ``[]`` rather than the un-filtered list. The
        whole point of the anchor is that we'd rather report no
        package than the wrong one (BMW-3454's bug was that the
        un-filtered fallback let Radioline's package through).
        """
        if not anchor or not candidates:
            return candidates
        return [c for c in candidates if self._in_anchor(c.value, anchor)]

    def _anchor_versions(
        self, candidates: List[_Candidate], anchor: set,
    ) -> List[_Candidate]:
        """Drop tagged version candidates whose embedded app/package
        is outside the summary's family.

        Three tag types we recognise on a version candidate:

          * ``catalog[App]:src`` / ``apps_template[App]:src`` — the
            embedded App must own a package in ``anchor``.
          * ``bmw_inventory[pkg]:src`` — the embedded package must
            itself be in (or a sub-variant of) ``anchor``. BMW
            CheckIn.txt versions are paired with their package
            on the same line, so the version is only legitimate if
            that package is the one we want.
          * No tag — labeled / phrase / NL-pair hits pass through
            untouched.

        Returns an empty list if every candidate is tagged and none
        of them belong to the summary family (better silent than
        wrong — BMW-3454 reproduced exactly this failure mode).
        """
        if not anchor or not candidates:
            return candidates
        keep: List[_Candidate] = []
        for c in candidates:
            inv_pkg = _bmw_inventory_pkg_from_label(c.source_label)
            if inv_pkg is not None:
                pkg_l = inv_pkg.lower()
                if pkg_l in anchor or any(
                    pkg_l.startswith(p + ".") for p in anchor
                ):
                    keep.append(c)
                continue
            cat_app = _catalog_app_from_label(c.source_label)
            if cat_app is None:
                keep.append(c)
                continue
            cat_entry = self._catalog.find_by_name(cat_app)
            if cat_entry is None:
                keep.append(c)
                continue
            if any(
                v.package and v.package.lower() in anchor
                for v in cat_entry.packages
            ):
                keep.append(c)
        return keep

    def _anchor_names(
        self, candidates: List[_Candidate], anchor: set,
    ) -> List[_Candidate]:
        """Same as :meth:`_anchor_versions` but for app-name
        candidates: drop catalog hits for apps that aren't in the
        summary's family. Free-text labeled hits (``App: …`` lines)
        pass through.
        """
        if not anchor or not candidates:
            return candidates
        keep: List[_Candidate] = []
        for c in candidates:
            cat_app = _catalog_app_from_label(c.source_label)
            if cat_app is None:
                keep.append(c)
                continue
            cat_entry = self._catalog.find_by_name(cat_app)
            if cat_entry is None:
                keep.append(c)
                continue
            if any(
                v.package and v.package.lower() in anchor
                for v in cat_entry.packages
            ):
                keep.append(c)
        return keep

    def _ver_candidate_consistent(
        self, candidate: _Candidate, final_app_name: Optional[str],
    ) -> bool:
        """Return True if ``candidate`` should survive the
        post-merge consistency filter.

        Catalog version candidates carry their app name in the
        ``catalog[App]:source`` label. If that name doesn't match
        ``final_app_name`` (case-insensitive), drop the candidate —
        otherwise we'd mix apps.
        Non-catalog candidates (labeled hits, NL pairs, anchored
        package-line patterns) pass through unchanged.
        """
        cat_app = _catalog_app_from_label(candidate.source_label)
        if cat_app is None:
            return True
        if final_app_name is None:
            # No app name decided yet — accept catalog hits only if a
            # single distinct catalog app appears across the pool.
            return True
        return cat_app.strip().lower() == final_app_name.strip().lower()

    def extract(self, issue: JiraIssue) -> Extraction:
        # ------------------------------------------------------------------
        # 1. Highest-trust source: Jira structured/custom fields
        # ------------------------------------------------------------------
        structured: StructuredFields = issue.structured or StructuredFields()

        log.warning(
            "[METADATA PIPELINE] %s | structured=%s | attachments=%d | "
            "description_len=%d | comments=%d",
            issue.key,
            structured,
            len(issue.attachments or []),
            len(issue.description or ""),
            len(issue.comments or []),
        )

        sources = _build_sources(issue)
        issue_anchor_tokens = _issue_app_anchor_tokens(sources)
        issue_anchor_names = _issue_app_anchor_names(sources)
        if issue_anchor_tokens:
            log.warning(
                "[ISSUE APP ANCHOR] %s | names=%s | tokens=%s",
                issue.key,
                issue_anchor_names,
                sorted(issue_anchor_tokens),
            )

        pkg_candidates = _scan_packages(sources)
        ver_candidates = _scan_versions(sources)
        name_candidates = _scan_app_names(sources)
        version_code_candidates = _scan_version_codes(sources)
        environment_candidates = _scan_environments(sources)
        inventory_pairs = _scan_bmw_inventory(sources)

        # ------------------------------------------------------------------
        # 2. Attachment/log boost
        # ------------------------------------------------------------------
        for att in issue.attachments:
            if not att.text:
                continue

            log.warning("[ATTACHMENT SCAN] %s | %s", issue.key, att.filename)

            attachment_source = _Source(
                label=f"attachment:{att.filename}",
                text=att.text,
                weight=4,
            )

            strong_pkg_patterns = [
                re.compile(
                    r"(?im)\b(?:Process|Package|packageName|applicationId)"
                    r"\s*[:=]\s*([a-z][a-z0-9_]+(?:\.[a-zA-Z0-9_]+){1,})"
                ),
                re.compile(
                    r"(?im)\b(?:crash due to|caused by|due to|in package)"
                    r"\s+([a-z][a-z0-9_]+(?:\.[a-zA-Z0-9_]+){1,})"
                ),
            ]

            for pattern in strong_pkg_patterns:
                for m in pattern.finditer(att.text):
                    pkg = m.group(1).strip()

                    if _is_acceptable_package(pkg):
                        pkg_candidates.append(_Candidate(
                            pkg,
                            _candidate_score(attachment_source, signal="phrase") + 5,
                            attachment_source.label,
                        ))

            strong_version_patterns = [
                _VERSION_FROM_PACKAGE_LINE_RE,
                _VERSION_FROM_VERSION_LINE_RE,
                _CRASH_ID_VERSION_RE,
                _VERSION_LABELED_RE,
            ]

            for pattern in strong_version_patterns:
                for m in pattern.finditer(att.text):
                    version = m.group(1).strip()

                    if version and version not in _VERSION_BLOCKLIST:
                        ver_candidates.append(_Candidate(
                            version,
                            _candidate_score(attachment_source, signal="phrase") + 5,
                            attachment_source.label,
                        ))

        # ------------------------------------------------------------------
        # 2b. Known-app catalog scan
        #
        # The catalog (bot/known_apps.json) is the bot's "common knowledge"
        # of apps — Zoom, Spotify, Deezer, Bild TV, Zoom for Cars, and so
        # on. It supplies the natural-language signal the bot was missing:
        # given "Error code 16 in zoom 1.0.8.9", recognise "zoom" as a
        # known app and pair it with the nearby "1.0.8.9".
        #
        # We treat catalog hits as phrase-strength evidence: the
        # combination of a known-app token AND a version within the
        # nearby window is far stronger than either alone.
        # ------------------------------------------------------------------
        for src in sources:
            for cmatch in self._catalog.find_in_text(src.text):
                base_score = _candidate_score(src, signal="phrase")
                # The summary is the ticket's headline — what the
                # reporter explicitly said the bug is about.
                if src.label == "summary":
                    base_score += 3
                # Embed the catalog app name in the source label so
                # the post-merge consistency check can verify that
                # name and version belong to the same app.
                catalog_label = f"catalog[{cmatch.app.name}]:{src.label}"
                # Use the canonical catalog name, not the matched alias
                # (so "ZOOM" / "zoom" still report as "Zoom").
                name_candidates.append(_Candidate(
                    cmatch.app.name,
                    base_score,
                    catalog_label,
                ))
                if cmatch.nearby_version \
                        and cmatch.nearby_version not in _VERSION_BLOCKLIST:
                    ver_candidates.append(_Candidate(
                        cmatch.nearby_version,
                        base_score,
                        catalog_label,
                    ))
                # Pick the right package variant for this match using
                # the version detected next to the alias.
                #
                # Catalog-derived package is a DERIVATIVE signal: we
                # know the app name and look up its registered
                # package. It must yield to any directly-observed
                # package (labeled ``packageName=X``, phrase
                # ``crash due to X``, NL pair, Jira custom field).
                # Otherwise the bot reports the catalog's generic
                # package even when the ticket explicitly names a
                # sub-variant — BMW-3455 lost
                # ``com.forvia.zoomapp.rsedemo`` to the catalog's
                # ``com.forvia.zoomapp`` exactly this way. The ``-4``
                # penalty puts catalog-derived packages below labeled
                # hits but still above bare scans.
                resolved_pkg = cmatch.resolved_package
                if resolved_pkg and _is_acceptable_package(resolved_pkg):
                    pkg_candidates.append(_Candidate(
                        resolved_pkg,
                        base_score - 4,
                        catalog_label,
                    ))

        # ------------------------------------------------------------------
        # 2b'. BMW STAT_APPS_TEXT inventory entries
        #
        # BMW CheckIn.txt attachments contain lines like
        # ``com.forvia.zoomapp.push_notifications;1.0.8.9RC11;270426 06:09``
        # which pair a package with its exact installed version on
        # the same line. This is the single most reliable signal for
        # version when the ticket has a CheckIn.txt — much better
        # than a catalog hit on an app name in passing prose.
        #
        # Both package and version are tagged with the SAME embedded
        # package so the summary anchor keeps them together (or drops
        # them together when the inventory entry isn't in the
        # summary-app family).
        # ------------------------------------------------------------------
        for inv_pkg, inv_ver, inv_src in inventory_pairs:
            label = f"bmw_inventory[{inv_pkg}]:{inv_src.label}"
            # Inventory is a package-version registry, not by itself proof
            # that the ticket is about this app. Keep the pair available,
            # but give it a conservative score; summary/description/crash
            # anchors and Admin Portal verification must decide.
            base_score = inv_src.weight + 2
            pkg_candidates.append(_Candidate(inv_pkg, base_score, label))
            ver_candidates.append(_Candidate(inv_ver, base_score, label))

        # Strong generic anchor: if the issue title/description names an
        # app that is not in the catalog, and CheckIn.txt contains an
        # inventory row whose package contains that app token, that row
        # becomes the leading candidate. This fixes cases such as:
        #   Summary: Bloomberg APP_...
        #   CheckIn: bloomberg.android.plus:1.0.2:130526 07:00
        # Without this, the bot can pick an unrelated installed app from
        # the same CheckIn inventory (e.g. Der Spiegel).
        if issue_anchor_tokens:
            for inv_pkg, inv_ver, inv_src in inventory_pairs:
                if not _package_matches_anchor_tokens(inv_pkg, issue_anchor_tokens):
                    continue
                label = f"issue_anchor_inventory[{inv_pkg}]:{inv_src.label}"
                base_score = inv_src.weight + 60
                pkg_candidates.append(_Candidate(inv_pkg, base_score, label))
                ver_candidates.append(_Candidate(inv_ver, base_score, label))
                anchored_name = _display_name_from_anchor_tokens(
                    issue_anchor_tokens, inv_pkg,
                )
                if anchored_name:
                    name_candidates.append(_Candidate(
                        anchored_name,
                        base_score,
                        label,
                    ))
                log.warning(
                    "[ISSUE ANCHOR INVENTORY MATCH] %s | %s:%s from %s",
                    issue.key,
                    inv_pkg,
                    inv_ver,
                    inv_src.label,
                )

        # ------------------------------------------------------------------
        # 2b''. Quoted app names following context cues
        #
        # Summary / description patterns like ``game "Unblock It"`` or
        # ``the app "Spotify"`` reveal the app even when neither the
        # catalog nor any package coordinate is in scope. We treat
        # these as high-trust signals because the human reporter
        # explicitly named the affected app.
        # ------------------------------------------------------------------
        for q_name, q_src in _scan_quoted_app_names(sources):
            base_score = q_src.weight + 7
            if q_src.label == "summary":
                base_score += 3
            label = f"catalog[{q_name}]:{q_src.label}"
            name_candidates.append(_Candidate(q_name, base_score, label))
            # If the catalog knows this app, also surface its
            # canonical package so the anchor and merge use it.
            cat_app = self._catalog.find_by_name(q_name)
            if cat_app:
                picked = cat_app.pick_package(None)
                if picked and _is_acceptable_package(picked):
                    pkg_candidates.append(_Candidate(
                        picked, base_score - 1, label,
                    ))

        # ------------------------------------------------------------------
        # 2c. BMW / Forvia "Apps: <name> v<version>" template
        #
        # This is the structured-description signal BMW reporters use:
        #   Apps (if specific version, please mention): Tagesschau Automotive v1.0.6
        # It's almost as strong as a Jira custom field — explicit,
        # intentional, and gives both the name and version. We score
        # it higher than a regular catalog hit (+8 vs +5) so the
        # specific name "Tagesschau Automotive" beats a passing
        # "Tagesschau" hit in the summary.
        # ------------------------------------------------------------------
        for tpl_name, tpl_ver, tpl_src in _scan_apps_template(sources):
            label = f"apps_template[{tpl_name}]:{tpl_src.label}"
            base_score = tpl_src.weight + 8
            name_candidates.append(_Candidate(tpl_name, base_score, label))
            if tpl_ver:
                ver_candidates.append(_Candidate(tpl_ver, base_score, label))
            # Catalog-reverse-lookup for the package so we don't have
            # to wait for the resolver step. ``pick_package`` lets a
            # version (when present) choose between multiple variants.
            cat_app = self._catalog.find_by_name(tpl_name)
            if cat_app:
                picked = cat_app.pick_package(tpl_ver)
                if picked and _is_acceptable_package(picked):
                    pkg_candidates.append(_Candidate(
                        picked, base_score - 1, label,
                    ))

        # ------------------------------------------------------------------
        # 3. Natural language pairs
        # ------------------------------------------------------------------
        for nl_name, nl_ver, nl_src in _scan_natural_language_pairs(sources):
            phrase_score = _candidate_score(nl_src, signal="phrase")

            name_candidates.append(_Candidate(
                nl_name,
                phrase_score,
                f"nl:{nl_src.label}",
            ))

            ver_candidates.append(_Candidate(
                nl_ver,
                phrase_score,
                f"nl:{nl_src.label}",
            ))

        # ------------------------------------------------------------------
        # 3b. Summary-app anchor
        #
        # The ticket summary is the reporter's headline — the explicit
        # statement of what the bug is about. When that headline names
        # a known catalog app (e.g. "Zoom: Entering passcode is not
        # intuitive"), the bot must NOT then pick a different app's
        # package from elsewhere in the ticket. BMW-3454 was failing
        # exactly that way: the summary clearly said "Zoom" but the
        # attached ``CheckIn.txt`` is a system-wide app inventory
        # listing dozens of apps (Radioline, Tagesschau, Spiegel,
        # Car Eats Car, …), and the catalog scan was happily picking
        # whichever inventory entry happened to have the strongest
        # nearby version.
        #
        # The anchor: if the summary names a known catalog app (or
        # any app in the same family — "Tagesschau" / "Tagesschau
        # Automotive", "Zoom" / "Zoom for Cars"), drop every package
        # candidate that isn't in that family's package set. Same for
        # versions and names. We only apply the anchor when filtering
        # actually keeps something — otherwise fall back to the
        # un-anchored pool.
        summary_app_packages = self._summary_anchor_packages(sources)
        if summary_app_packages:
            log.info(
                "[SUMMARY ANCHOR] %s | %d package(s) in summary-app family",
                issue.key, len(summary_app_packages),
            )
            pkg_candidates = self._anchor_packages(
                pkg_candidates, summary_app_packages,
            )
            ver_candidates = self._anchor_versions(
                ver_candidates, summary_app_packages,
            )
            name_candidates = self._anchor_names(
                name_candidates, summary_app_packages,
            )

        regex_pkg = _best(pkg_candidates)
        regex_ver = _best(ver_candidates)
        regex_name = _best(name_candidates)

        # Guardrail: a BMW inventory-only package is weak unless the
        # issue text also anchors the app family. Inventory lists every
        # installed app, so it must not decide the affected app alone.
        inventory_only_package = bool(
            regex_pkg
            and regex_pkg.source_label.startswith("bmw_inventory[")
            and not summary_app_packages
            and not structured.package_name
            and not structured.app_name
        )
        if inventory_only_package:
            log.warning(
                "[INVENTORY ONLY GUARD] %s | dropping package candidate %s "
                "because CheckIn inventory is not enough without a "
                "summary/description/Jira-field anchor",
                issue.key, regex_pkg.value,
            )
            regex_pkg = None

        # ------------------------------------------------------------------
        # 4. Merge priority
        #
        # Priority:
        # 1. Jira structured/custom fields
        # 2. Admin Portal / resolver validation
        # 3. Attachments/logs
        # 4. Description/comments
        # 5. Summary regex fallback
        # ------------------------------------------------------------------

        def _dedup(values: List[str]) -> List[str]:
            seen = set()
            out: List[str] = []

            for value in values:
                if not value:
                    continue

                cleaned = str(value).strip()

                if not cleaned:
                    continue

                if cleaned.lower() in seen:
                    continue

                seen.add(cleaned.lower())
                out.append(cleaned)

            return out

        final_package = structured.package_name or (
            regex_pkg.value if regex_pkg else None
        )

        # ----- App name selection (package-anchored)
        #
        # The bug we're fixing here: BMW-3401 had its package correctly
        # identified as com.radioline.android.radioline.auto, but the
        # bot reported app_name="YouTube" because YouTube was mentioned
        # in an attached log alongside another version. The catalog
        # name candidate must be CONSISTENT with the identified
        # package — otherwise we end up reporting metadata that
        # belongs to two different apps. Same idea for versions.
        # ``name_source`` is a one-line label recording where the app
        # name came from — drives the ``evidence['app_name']`` value
        # in the final Extraction so users can audit the decision.
        name_source: str = ""
        if structured.app_name:
            final_app_name = structured.app_name
            name_source = "jira_structured_field"
        elif final_package:
            # Step 1: prefer a regex name candidate that's CONSISTENT
            # with the chosen package (catalog hit pointing at the
            # SAME package family). Keeps the most-specific signal —
            # e.g. "Zoom for Cars 1.0.8.7" beats the generic "Zoom".
            if regex_name and self._name_consistent_with_package(
                regex_name.value, final_package,
            ):
                final_app_name = regex_name.value
                name_source = regex_name.source_label
            else:
                # Step 2: catalog reverse-lookup by package — picks
                # up apps that are in the catalog but weren't
                # mentioned by name in the ticket text (e.g.
                # com.radioline.android.radioline.auto → Radioline).
                cat_app_for_pkg = self._catalog.find_by_package(final_package)
                if cat_app_for_pkg:
                    final_app_name = cat_app_for_pkg.name
                    name_source = f"catalog_by_package:{final_package}"
                else:
                    # Step 3: derive from the package coordinate.
                    # Conservative — only fires when a segment
                    # actually repeats.
                    derived = _derive_name_from_package(final_package)
                    if derived:
                        final_app_name = derived
                        name_source = f"derived_from_package:{final_package}"
                    else:
                        # Catalog name disagreed with package AND we
                        # can't derive — drop rather than misattribute.
                        final_app_name = None
        elif regex_name:
            final_app_name = regex_name.value
            name_source = regex_name.source_label
        else:
            final_app_name = None

        # ----- Version selection (filter inconsistent catalog hits)
        final_versions: List[str] = []

        if structured.version_names:
            final_versions.extend(structured.version_names)

        if structured.app_fix_versions:
            final_versions.extend(structured.app_fix_versions)

        final_versions = _dedup(final_versions)

        version_source: str = ""

        if not final_versions and final_package:
            # Highest-trust version fallback: if the chosen package came
            # from BMW CheckIn inventory, take ONLY the version from the
            # SAME inventory entry. Never combine package from app A with
            # version from app B.
            paired_inventory_versions = [
                _Candidate(inv_ver, inv_src.weight + 6,
                           f"bmw_inventory[{inv_pkg}]:{inv_src.label}")
                for inv_pkg, inv_ver, inv_src in inventory_pairs
                if inv_pkg.strip().lower() == final_package.strip().lower()
            ]
            best_inventory_version = _best(paired_inventory_versions)
            if best_inventory_version is not None:
                final_versions = [best_inventory_version.value]
                version_source = best_inventory_version.source_label

        if not final_versions:
            # Drop catalog ver candidates whose embedded app name
            # disagrees with our chosen final_app_name. This kills the
            # "YouTube 2.0.68 from a side mention" pattern that bled
            # into BMW-3401's notification.
            consistent_ver_candidates = [
                c for c in ver_candidates
                if self._ver_candidate_consistent(c, final_app_name)
            ]
            best_filtered = _best(consistent_ver_candidates)
            if best_filtered is not None:
                final_versions = [best_filtered.value]
                version_source = best_filtered.source_label

        final_version_code = structured.version_code or (
            _best(version_code_candidates).value
            if _best(version_code_candidates) else None
        )
        final_environment_name = structured.environment_name or (
            _best(environment_candidates).value
            if _best(environment_candidates) else None
        )

        # ------------------------------------------------------------------
        # 5. Package/app resolver
        #
        # Bidirectional now:
        #   * package known, name missing  → resolver.resolve(pkg)
        #   * name known, package missing  → resolver.resolve_package_by_name
        #     (catalog first, then Admin Portal)
        #   * package known, no version    → resolver.resolve_versions(pkg)
        # ------------------------------------------------------------------
        name_from_map = False
        package_from_resolver = False
        admin_verified = False

        # Reverse lookup: detected app name with no package yet.
        if final_app_name and not final_package:
            # Catalog reverse-lookup is local and free. Use the first
            # detected version (if any) to disambiguate among multiple
            # variants of the same app — e.g. "Zoom" + "1.0.8.9" picks
            # com.forvia.zoomapp, while "Zoom" + "5.6.0" picks
            # us.zoom.videomeetings.
            cat_app = self._catalog.find_by_name(final_app_name)
            picked_pkg = None
            if cat_app:
                version_hint = (
                    final_versions[0] if final_versions else None
                )
                picked_pkg = cat_app.pick_package(version_hint)
            if picked_pkg and _is_acceptable_package(picked_pkg):
                final_package = picked_pkg
                package_from_resolver = True
            else:
                try:
                    portal_pkg = self._app_name_resolver.resolve_package_by_name(
                        final_app_name,
                    )
                except Exception as exc:
                    log.warning(
                        "Resolver.resolve_package_by_name failed for %s: %s",
                        final_app_name, exc,
                    )
                    portal_pkg = None
                if portal_pkg and _is_acceptable_package(portal_pkg):
                    final_package = portal_pkg
                    package_from_resolver = True

        if final_package:
            if not final_app_name:
                try:
                    mapped = self._app_name_resolver.resolve(final_package)
                except Exception as exc:
                    log.warning(
                        "App-name resolver failed for %s: %s",
                        final_package,
                        exc,
                    )
                    mapped = None

                if mapped:
                    final_app_name = mapped
                    name_from_map = True

            # NOTE: We deliberately do NOT fall back to
            # ``resolve_versions(package)`` to fill in the version
            # field. That would dump the portal's WHOLE published-
            # versions list into the answer and the Slack notification
            # would claim the ticket was filed against every approved
            # build at once. Under the strict mandate, the version
            # field stays empty when no candidate was extracted from
            # the ticket itself — the portal list is then surfaced
            # separately as an "Admin Portal: Approved versions: ..."
            # line so the assignee can identify the right build by
            # hand.

        # ------------------------------------------------------------------
        # 6. Admin Portal validation — STRICT GATE
        #
        # The Admin Portal is the back-office source of truth. The bot
        # may only publish a "high confidence" Slack banner when:
        #
        #   1. We have a package coordinate, AND
        #   2. That package exists in the Admin Portal, AND
        #   3. The extracted version is on the portal's approved list.
        #
        # When (2) fails (package not in portal) the bot is in pure-
        # guess territory — confidence drops to medium so the assignee
        # is asked to confirm manually.
        #
        # When (3) fails (package present, version not on the approved
        # list) the bot still trusts the portal's app name and surfaces
        # the version mismatch in evidence — it could legitimately be
        # a brand-new build under test, but it's not a "verified" state.
        #
        # When the configured resolver simply has nothing to say (no
        # snapshot, no API, transient failure) we fall back to legacy
        # ``not_checked`` behaviour rather than refusing to post — the
        # bot is still better than nothing.
        # ------------------------------------------------------------------
        admin_portal_status = "not_checked"
        admin_portal_versions: List[str] = []
        resolve_full = getattr(self._app_name_resolver, "resolve_full", None)
        # Only portal-capable resolvers should trip the strict gate.
        # A plain JsonFileAppNameResolver doesn't speak for the portal —
        # treating its empty answer as ``no_match`` would downgrade
        # every legacy run that hasn't configured a snapshot.
        provides_verification = getattr(
            self._app_name_resolver, "provides_admin_verification", False,
        )

        if callable(resolve_full) and provides_verification \
                and (final_package or final_app_name):
            try:
                log.warning(
                    "[ADMIN VALIDATION] %s | package=%s | app_name=%s",
                    issue.key,
                    final_package,
                    final_app_name,
                )

                # STRICT PORTAL GATE:
                # If we already have a package candidate, validate ONLY by package.
                # Never pass app_name together with package, otherwise the resolver may
                # fall back to the app-name hit and incorrectly mark a mismatched
                # package/app pair as verified.
                validation_package = final_package
                validation_app_name = None if final_package else final_app_name

                try:
                    verified = resolve_full(
                        package=validation_package,
                        app_name=validation_app_name,
                    )
                except TypeError:
                    verified = resolve_full(
                        package_name=validation_package,
                        app_name=validation_app_name,
                    )

                if verified:
                    if isinstance(verified, dict):
                        verified_app = (
                            verified.get("app_name")
                            or verified.get("name")
                            or verified.get("application_name")
                        )
                        verified_pkg = (
                            verified.get("package_name")
                            or verified.get("package")
                            or verified.get("application_id")
                            or verified.get("applicationId")
                        )
                        verified_versions = (
                            verified.get("versions")
                            or verified.get("app_versions")
                            or verified.get("version_names")
                            or verified.get("versionName")
                            or verified.get("app_version")
                        )
                    else:
                        verified_app = getattr(verified, "app_name", None)
                        verified_pkg = getattr(verified, "package_name", None)
                        verified_versions = (
                            getattr(verified, "versions", None)
                            or getattr(verified, "app_versions", None)
                            or getattr(verified, "version_names", None)
                            or getattr(verified, "app_version", None)
                        )

                    # Portal is authoritative on the app name: rename
                    # "Streamingmedia" → "Radio Format" etc.
                    if verified_app and _is_acceptable_app_name(str(verified_app)):
                        if final_app_name and str(verified_app).strip().lower() \
                                != final_app_name.strip().lower():
                            log.warning(
                                "[ADMIN VALIDATION RENAME] %s | "
                                "%s → %s (portal authoritative)",
                                issue.key, final_app_name, verified_app,
                            )
                        final_app_name = str(verified_app).strip()

                    if verified_pkg and _is_acceptable_package(str(verified_pkg)):
                        final_package = str(verified_pkg).strip()

                    # Normalise the portal's published-version list.
                    if verified_versions:
                        if isinstance(verified_versions, str):
                            admin_portal_versions = _dedup(re.split(
                                r"[,;/]| and | & |\n|\r|\s\|\s",
                                verified_versions,
                            ))
                        elif isinstance(verified_versions, list):
                            admin_portal_versions = _dedup([
                                str(v) for v in verified_versions if v
                            ])

                    # Decide verified vs. version_mismatch by comparing
                    # extracted version against the portal's approved
                    # list. We accept exact match OR portal-prefix
                    # match (e.g. extracted "1.0.8.9RC13" matches
                    # portal entry "1.0.8.9" because RC suffixes are
                    # build-level metadata).
                    extracted_versions = list(final_versions)
                    portal_versions = list(admin_portal_versions)

                    def _version_present(v: str) -> bool:
                        v_norm = v.strip().lower()
                        for pv in portal_versions:
                            pv_norm = pv.strip().lower()
                            if v_norm == pv_norm:
                                return True
                            if v_norm.startswith(pv_norm + "rc"):
                                return True
                            if pv_norm.startswith(v_norm + "rc"):
                                return True
                        return False

                    if not portal_versions:
                        # Portal entry exists but no version list is on
                        # file (snapshot row may be partial). Treat as
                        # verified-on-package only.
                        admin_portal_status = "verified"
                    elif not extracted_versions:
                        # No version was extracted — the portal does
                        # know the package though, so flag as a
                        # version_mismatch (needs manual confirmation).
                        admin_portal_status = "version_mismatch"
                    elif all(_version_present(v) for v in extracted_versions):
                        admin_portal_status = "verified"
                        # Snap the extracted version to the portal's
                        # exact string when we matched via the RC-
                        # prefix rule, so the Slack notification shows
                        # the same string the portal does.
                        normalised: List[str] = []
                        for v in extracted_versions:
                            matched = next(
                                (pv for pv in portal_versions
                                 if pv.strip().lower() == v.strip().lower()),
                                None,
                            )
                            normalised.append(matched if matched else v)
                        final_versions = _dedup(normalised) or extracted_versions
                    else:
                        # ----------------------------------------------
                        # version_mismatch RESCUE
                        #
                        # The "best" version candidate doesn't match
                        # the portal's published list — but the ticket
                        # text may carry SEVERAL version strings (head-
                        # unit build IDs, log timestamps, app build
                        # numbers). If ANY of the other candidates is
                        # on the portal list, use that one. This is the
                        # BMW-3450 case: bot picked the head-unit's
                        # "2.2607.8-POINTFIX" but the ticket also
                        # mentioned the actual Zoom build buried in a
                        # comment / attachment.
                        # ----------------------------------------------
                        rescued: Optional[str] = None
                        for cand in ver_candidates:
                            cand_value = (cand.value or "").strip()
                            if not cand_value:
                                continue
                            if _version_present(cand_value):
                                # Snap to the portal's exact string.
                                matched = next(
                                    (pv for pv in portal_versions
                                     if pv.strip().lower()
                                     == cand_value.strip().lower()),
                                    cand_value,
                                )
                                rescued = matched
                                break
                        if rescued:
                            log.warning(
                                "[ADMIN VALIDATION RESCUE] %s | "
                                "extracted %s not on portal list — "
                                "swapping in candidate %s which IS on "
                                "the list",
                                issue.key, extracted_versions, rescued,
                            )
                            final_versions = [rescued]
                            admin_portal_status = "verified"
                        else:
                            # No candidate matched. The bot must NOT
                            # publish "App Version: 2.2607.8-POINTFIX"
                            # because that string is not in the
                            # Admin Portal — the user has been
                            # explicit: no answers unless they're
                            # 100% aligned. Suppress the version.
                            log.warning(
                                "[ADMIN VALIDATION VERSION SUPPRESSED] "
                                "%s | extracted %s NOT on portal list "
                                "%s — clearing version field; the "
                                "Slack note will surface the portal's "
                                "approved versions instead.",
                                issue.key, extracted_versions,
                                portal_versions,
                            )
                            final_versions = []
                            admin_portal_status = "version_mismatch"

                    admin_verified = admin_portal_status == "verified"
                    log.warning(
                        "[ADMIN VALIDATION RESULT] %s | status=%s | "
                        "portal_versions=%s",
                        issue.key, admin_portal_status, portal_versions,
                    )
                else:
                    # Portal returned no record for this package — the
                    # bot is on its own. Downgrade unless we have
                    # nothing to validate (no package extracted).
                    if final_package:
                        admin_portal_status = "no_match"
                        log.warning(
                            "[ADMIN VALIDATION NO_MATCH] %s | package=%s "
                            "not found in Admin Portal — confidence "
                            "will be downgraded",
                            issue.key, final_package,
                        )
                    else:
                        admin_portal_status = "not_checked"

            except Exception as exc:
                log.warning(
                    "[ADMIN VALIDATION FAILED] %s | %s",
                    issue.key,
                    exc,
                )
                admin_portal_status = "not_checked"

        # ------------------------------------------------------------------
        # 7. Confidence
        #
        # A name derived from the package coordinate is a HEURISTIC,
        # not a verified mapping — confidence must be capped at
        # "medium" so the Slack notification asks for manual
        # validation. The user explicitly called this out on
        # BMW-3438: "Automotive" was derived from
        # ``de.spiegel.android.automotive.mmo`` and reported with
        # high confidence, which was wrong.
        # ------------------------------------------------------------------
        name_is_derived = name_source.startswith("derived_from_package:")
        if admin_verified and final_package and final_app_name:
            confidence = "high"
        elif structured.has_any() and final_app_name and final_package and final_versions:
            # Structured fields are strong, but the bot may only claim
            # "high" without an Admin Portal verdict when the portal
            # was simply not consulted (legacy mode). When the portal
            # WAS consulted and answered "no_match" or
            # "version_mismatch", we must downgrade.
            if admin_portal_status in ("no_match", "version_mismatch"):
                confidence = "medium"
            else:
                confidence = "high"
        else:
            confidence = _compute_confidence_v3(
                has_pkg=bool(final_package),
                has_name=bool(final_app_name),
                has_ver=bool(final_versions),
                structured=structured,
                name_from_map=name_from_map,
                regex_pkg=regex_pkg,
                regex_ver=regex_ver,
                regex_name=regex_name,
                strong_threshold=self._STRONG_SCORE,
            )
            # Portal disagrees with what the bot has — downgrade.
            if admin_portal_status in ("no_match", "version_mismatch") \
                    and confidence == "high":
                confidence = "medium"
        # Final cap: a derived name is never high-confidence.
        if name_is_derived and confidence == "high":
            confidence = "medium"

        # ------------------------------------------------------------------
        # 8. Evidence
        # ------------------------------------------------------------------
        evidence = {
            # ``name_source`` is set above by the package-anchored
            # selector; fall back to the legacy helper for the
            # no-package path.
            "app_name": (
                name_source if name_source else _evidence_for(
                    "app_name",
                    structured_value=structured.app_name,
                    regex_candidate=regex_name,
                    from_map=name_from_map,
                    package=final_package,
                )
            ),
            "app_version": (
                version_source if version_source else _evidence_for(
                    "app_version",
                    structured_value=(
                        ", ".join(structured.version_names)
                        if structured.version_names else None
                    ),
                    regex_candidate=regex_ver,
                    from_map=False,
                )
            ),
            "version_code": _evidence_for(
                "version_code",
                structured_value=structured.version_code,
                regex_candidate=_best(version_code_candidates),
                from_map=False,
            ),
            "environment_name": _evidence_for(
                "environment_name",
                structured_value=structured.environment_name,
                regex_candidate=_best(environment_candidates),
                from_map=False,
            ),
            "package_name": _evidence_for(
                "package_name",
                structured_value=structured.package_name,
                regex_candidate=regex_pkg,
                from_map=False,
            ),
        }

        # Always surface the portal status — the assignee should see at
        # a glance whether the bot's answer was independently confirmed.
        evidence["admin_portal"] = admin_portal_status
        if admin_portal_status == "verified" and admin_portal_versions:
            evidence["admin_portal_versions"] = ", ".join(admin_portal_versions)

        backend_label = self.name

        if structured.has_any():
            backend_label = "regex+structured"

        if admin_verified:
            backend_label = f"{backend_label}+admin_verified"
        elif admin_portal_status == "no_match":
            backend_label = f"{backend_label}+portal_no_match"
        elif admin_portal_status == "version_mismatch":
            backend_label = f"{backend_label}+portal_version_mismatch"

        ext = Extraction(
            app_name=final_app_name,
            app_version=", ".join(final_versions) if final_versions else None,
            app_versions=final_versions,
            package_name=final_package,
            version_code=final_version_code,
            environment_name=final_environment_name,
            confidence=confidence,
            evidence=evidence,
            backend=backend_label,
            admin_portal_status=admin_portal_status,
            admin_portal_versions=admin_portal_versions,
        )

        log.warning("FINAL EXTRACTION for %s: %s", issue.key, ext.to_dict())
        return ext


def _compute_confidence_v3(
    *, has_pkg: bool, has_name: bool, has_ver: bool,
    structured: StructuredFields,
    name_from_map: bool,
    regex_pkg: Optional[_Candidate],
    regex_ver: Optional[_Candidate],
    regex_name: Optional[_Candidate],
    strong_threshold: int,
) -> str:
    structured_pkg = bool(structured.package_name)
    structured_name = bool(structured.app_name)
    structured_ver = bool(structured.version_names)

    strong_pkg = (
        structured_pkg
        or (regex_pkg is not None and regex_pkg.score >= strong_threshold)
    )
    strong_ver = (
        structured_ver
        or (regex_ver is not None and regex_ver.score >= strong_threshold)
    )
    strong_name = (
        structured_name
        or name_from_map
        or (regex_name is not None and regex_name.score >= strong_threshold)
    )

    if has_pkg and has_ver and has_name:
        strong_count = int(strong_pkg) + int(strong_ver) + int(strong_name)

        if strong_count >= 2:
            return "high"

        return "medium"

    if has_pkg and (has_ver or has_name):
        return "medium"

    if has_pkg:
        return "medium"

    if structured.has_any():
        return "medium"

    if strong_ver and strong_name:
        return "medium"

    return "low"


def _evidence_for(
    field: str, *,
    structured_value: Optional[str],
    regex_candidate: Optional[_Candidate],
    from_map: bool,
    package: Optional[str] = None,
) -> str:
    if structured_value:
        return "jira_structured_field"

    if from_map and package:
        return f"package_app_map:{package}"

    if regex_candidate is not None:
        return regex_candidate.source_label

    return ""


_CLAUDE_SYSTEM_PROMPT = (
    "You extract Android app metadata from Jira issues for an "
    "automotive QA team. Use ONLY information that is explicitly "
    "present in the input. Never invent values. If a field cannot be "
    "confidently identified, return null for that field. "
    "Output strictly valid JSON, nothing else."
)


_CLAUDE_OUTPUT_SHAPE = {
    "app_name": "string or null",
    "app_version": "string or null",
    "package_name": "string or null",
    "version_code": "string or null",
    "environment_name": "string or null",
    "confidence": "one of: low, medium, high",
    "rationale": "short string (<= 200 chars) explaining the choice",
}


class ClaudeExtractor(ExtractorBackend):
    name = "claude"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5"):
        from anthropic import Anthropic  # type: ignore

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def extract(self, issue: JiraIssue) -> Extraction:
        prompt = self._build_prompt(issue)

        try:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=_CLAUDE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            log.warning("Claude call failed for %s: %s", issue.key, exc)
            return Extraction(backend=self.name)

        text = ""

        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", "") == "text":
                text += getattr(block, "text", "")

        parsed = _parse_strict_json(text)

        if not parsed:
            log.warning(
                "Claude returned non-JSON for %s: %r",
                issue.key,
                text[:200],
            )
            return Extraction(backend=self.name)

        return Extraction(
            app_name=_clean(parsed.get("app_name")),
            app_version=_clean(parsed.get("app_version")),
            package_name=_clean(parsed.get("package_name")),
            version_code=_clean(parsed.get("version_code")),
            environment_name=_clean(parsed.get("environment_name")),
            confidence=str(parsed.get("confidence") or "low").lower()
            if parsed.get("confidence") in CONFIDENCE_LEVELS
            else "low",
            evidence={
                "rationale": str(parsed.get("rationale") or "")[:200],
            },
            backend=self.name,
        )

    @staticmethod
    def _build_prompt(issue: JiraIssue) -> str:
        parts: List[str] = []

        parts.append(f"## Issue\nKey: {issue.key}\nSummary: {issue.summary}")

        if issue.description:
            parts.append(
                "## Description\n" + truncate(issue.description, 2_000)
            )

        if issue.comments:
            parts.append("## Comments")

            for c in issue.comments[:8]:
                parts.append(
                    f"- ({c.author}, {c.created}) {truncate(c.body, 600)}"
                )

        for att in issue.attachments:
            if att.text:
                parts.append(
                    f"## Attachment: {att.filename}\n"
                    + truncate(att.text, 4_000)
                )

        parts.append(
            "## Required output\n"
            "Return JSON exactly matching this shape (no markdown fences):\n"
            + json.dumps(_CLAUDE_OUTPUT_SHAPE, indent=2)
        )

        return "\n\n".join(parts)


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None

    s = str(value).strip()

    if not s or s.lower() in {"null", "none", "n/a", "unknown"}:
        return None

    return s


def _parse_strict_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if not m:
            return None

        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


class HybridExtractor:
    def __init__(self, *, claude: Optional[ClaudeExtractor] = None) -> None:
        self._regex = RegexExtractor()
        self._claude = claude

    def extract(self, issue: JiraIssue) -> Extraction:
        primary = self._regex.extract(issue)

        if self._claude is None:
            return primary

        if primary.confidence != "low" and not primary.missing_fields():
            return primary

        log.info(
            "Refining %s via Claude (regex confidence=%s, missing=%s)",
            issue.key,
            primary.confidence,
            primary.missing_fields(),
        )

        secondary = self._claude.extract(issue)
        return _merge(primary, secondary)


def _merge(primary: Extraction, secondary: Extraction) -> Extraction:
    fields = ("app_name", "app_version", "package_name", "version_code", "environment_name")
    merged_values: Dict[str, Optional[str]] = {}
    evidence = dict(primary.evidence)

    for f in fields:
        p_val = getattr(primary, f)
        s_val = getattr(secondary, f)

        if p_val:
            merged_values[f] = p_val
        elif s_val:
            merged_values[f] = s_val
            evidence[f] = "claude:" + (
                secondary.evidence.get("rationale", "") or "fallback"
            )
        else:
            merged_values[f] = None

    levels = {"low": 0, "medium": 1, "high": 2}
    p_lvl = levels.get(primary.confidence, 0)
    s_lvl = levels.get(secondary.confidence, 0)
    final_lvl_idx = max(p_lvl, s_lvl) if any(merged_values.values()) else 0
    final = ["low", "medium", "high"][final_lvl_idx]

    if primary.app_version and primary.app_versions:
        merged_versions = list(primary.app_versions)
    elif (not primary.app_version) and secondary.app_versions:
        merged_versions = list(secondary.app_versions)
    elif merged_values["app_version"]:
        merged_versions = [merged_values["app_version"]]
    else:
        merged_versions = []

    return Extraction(
        app_name=merged_values["app_name"],
        app_version=merged_values["app_version"],
        app_versions=merged_versions,
        package_name=merged_values["package_name"],
        version_code=merged_values["version_code"],
        environment_name=merged_values["environment_name"],
        confidence=final,
        evidence=evidence,
        backend=f"{primary.backend}+{secondary.backend}",
    )