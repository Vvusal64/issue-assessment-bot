"""Known-app catalog with bidirectional lookup, NL search, and
version-aware package disambiguation.

Why version-aware? Many apps have several published packages — the
"Zoom" family alone has at least three:

  * ``com.forvia.zoomapp``      → Forvia "Zoom for Cars" (v1.0.x family)
  * ``us.zoom.videomeetings``   → Stellantis Zoom        (v5.x family)
  * ``us.zoom.mercedesbenz``    → Zoom for Mercedes      (v6.7.x / 7.0.x)

When a Jira ticket says "zoom 1.0.8.9", we want the bot to pick
``com.forvia.zoomapp`` based on the version prefix — picking the
"first known Zoom" would be wrong. The catalog handles this with
``KnownApp.pick_package(version)``: each variant carries a list of
``version_prefixes`` and the picker returns the variant whose prefix
matches the detected version.

Catalog file format (``bot/known_apps.json``)::

    {
      "name": "Zoom",
      "aliases": [],
      "packages": [
        {"package": "com.forvia.zoomapp",   "version_prefixes": ["1.0.", "1.1."]},
        {"package": "us.zoom.videomeetings", "version_prefixes": ["5."]},
        {"package": "us.zoom.mercedesbenz",  "version_prefixes": ["6.7.", "7.0."]}
      ]
    }

The legacy single-package form is still accepted::

    {"name": "Spotify", "package": "com.spotify.music"}
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


log = logging.getLogger("bot.catalog")


_DEFAULT_CATALOG_PATH = os.path.join(
    os.path.dirname(__file__), "known_apps.json",
)


# Version token used when scanning near a known-app mention.
_NEAR_VERSION_RE = re.compile(
    r"\b(?:v\s*)?"
    r"(\d+\.\d+\.\d+(?:\.\d+)?(?:[-_+.][A-Za-z0-9]+)?)"
    r"\b"
)

# Window (in characters) around the matched alias in which we'll
# accept a version token as belonging to the same mention.
_NEAR_VERSION_WINDOW = 80

# Cue words that signal a nearby name is an app/game, not a common
# English word. These must be the word IMMEDIATELY before or after
# the alias — looking further afield falsely accepts the bare
# adjective "audible" just because "the" appears earlier in the
# sentence.
_CONTEXT_CUE_WORDS = frozenset({
    "app", "apps", "application", "applications", "game", "games",
    "open", "launch", "install", "update", "crash", "crashes",
    "start", "opened", "launched", "installed", "published",
    "packagename", "package", "pkg", "process",
})
_DEFINITE_ARTICLES = frozenset({"the", "der", "die", "das"})
_TRAILING_WORD_RE = re.compile(r"(\w+)\s*$")
_LEADING_WORD_RE = re.compile(r"^\s*(\w+)")


def _has_context_cue(text: str, start: int, length: int) -> bool:
    """Return True if the alias match at ``[start, start+length)`` is
    framed by a context cue:

      * a strong punctuation marker (colon after, quotes around), or
      * a cue word IMMEDIATELY before or after the alias, or
      * a "the/der/die/das X app/application/game" pattern.

    "the Audible app" → accept (cue "app" after).
    "Audible: cannot login" → accept (colon after).
    "IDCevo audible from the next room" → reject (no immediate cue).
    """
    end = start + length

    # 1. Strong adjacent punctuation: "AppName:" or quotes.
    after_char = text[end] if end < len(text) else ""
    before_char = text[start - 1] if start > 0 else ""
    if after_char == ":":
        return True
    if before_char in "\"'`«‘“" and after_char in "\"'`»’”´":
        return True

    # 2. Immediately adjacent words.
    word_before_m = _TRAILING_WORD_RE.search(text[:start])
    word_before = word_before_m.group(1).lower() if word_before_m else ""
    word_after_m = _LEADING_WORD_RE.match(text[end:])
    word_after = word_after_m.group(1).lower() if word_after_m else ""

    if word_before in _CONTEXT_CUE_WORDS or word_after in _CONTEXT_CUE_WORDS:
        return True

    # 3. "the/der X app/application/game" sandwich.
    if word_before in _DEFINITE_ARTICLES \
            and word_after in {"app", "application", "game"}:
        return True

    return False


@dataclass
class PackageVariant:
    """One package coordinate that an app can be published under.

    ``version_prefixes`` is the set of version-string prefixes that
    uniquely identify this variant. Empty list = "default fallback"
    (used when the entry is single-package or when no other variant
    matches).
    """
    package: str
    version_prefixes: List[str] = field(default_factory=list)

    def matches_version(self, version: str) -> bool:
        if not self.version_prefixes:
            return False
        return any(version.startswith(p) for p in self.version_prefixes)


@dataclass
class KnownApp:
    """A single catalog entry.

    ``requires_context`` is for apps whose name is also a common
    English/German word (Audible "able to be heard", Maps,
    Pages, …). When True, ``KnownAppCatalog.find_in_text`` only
    accepts the match if the alias appears next to a context cue —
    "app", "application", "game", "the X", "X app", ``open X`` …
    """
    name: str
    aliases: List[str] = field(default_factory=list)
    packages: List[PackageVariant] = field(default_factory=list)
    requires_context: bool = False

    def all_names(self) -> List[str]:
        return [self.name, *self.aliases]

    @property
    def primary_package(self) -> Optional[str]:
        """First listed variant's package — used as a presentation
        hint in places that don't have a version yet."""
        return self.packages[0].package if self.packages else None

    @property
    def package(self) -> Optional[str]:
        """Backward-compat accessor identical to :attr:`primary_package`."""
        return self.primary_package

    def pick_package(self, version: Optional[str]) -> Optional[str]:
        """Return the package coordinate for this app given the
        ``version`` we observed nearby.

        Rules:
          * Zero variants → ``None``.
          * Exactly one variant → return it (regardless of version).
          * Multiple variants and a version is supplied → return the
            variant whose ``version_prefixes`` matches the longest
            prefix of ``version``. If multiple variants share the
            same top score, the first listed in the catalog wins —
            ordering is hand-curated to put the production/canonical
            variant first (e.g. ``push_notifications`` before the
            ``rsedemo`` demo variant of Zoom for Cars).
          * Multiple variants and no version (or no match) → return a
            single variant with empty prefixes (the explicit default),
            otherwise ``None``. We refuse to guess between ambiguous
            packages — better silent than wrong.
        """
        if not self.packages:
            return None
        if len(self.packages) == 1:
            return self.packages[0].package

        if version:
            scored: List = []
            for idx, v in enumerate(self.packages):
                # Score = length of the longest matching prefix
                # (more specific = higher).
                best = max(
                    (len(p) for p in v.version_prefixes
                     if version.startswith(p)),
                    default=0,
                )
                if best > 0:
                    scored.append((best, idx, v))
            if scored:
                # Highest score wins; on tie, prefer the variant
                # listed first in the catalog (lowest idx). The
                # catalog is hand-ordered: production/primary first,
                # demo/alternate variants after.
                scored.sort(key=lambda t: (-t[0], t[1]))
                return scored[0][2].package

        # No version, or no prefix matched: fall back to the variant
        # explicitly marked as default (empty prefixes).
        defaults = [v for v in self.packages if not v.version_prefixes]
        if len(defaults) == 1:
            return defaults[0].package
        return None


@dataclass
class CatalogMatch:
    """One ``find_in_text`` hit."""
    app: KnownApp
    matched_alias: str
    position: int
    nearby_version: Optional[str] = None

    @property
    def resolved_package(self) -> Optional[str]:
        """Return the package coordinate for this match, picked using
        the ``nearby_version`` if any. ``None`` if the catalog can't
        confidently disambiguate without an Admin Portal lookup."""
        return self.app.pick_package(self.nearby_version)


class KnownAppCatalog:
    """In-memory catalog with whole-word, longest-alias-first matching."""

    def __init__(self, apps: List[KnownApp]) -> None:
        self._apps: List[KnownApp] = list(apps)
        self._by_package: Dict[str, KnownApp] = {}
        self._by_alias_lower: Dict[str, KnownApp] = {}
        for app in self._apps:
            for variant in app.packages:
                self._by_package.setdefault(
                    variant.package.lower(), app,
                )
            for alias in app.all_names():
                self._by_alias_lower.setdefault(alias.lower(), app)
        self._aliases_by_length: List[str] = sorted(
            self._by_alias_lower.keys(), key=len, reverse=True,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load_default(cls) -> "KnownAppCatalog":
        return cls.load(_DEFAULT_CATALOG_PATH)

    @classmethod
    def load(cls, path: str) -> "KnownAppCatalog":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            log.warning(
                "Could not load catalog from %s: %s — using empty catalog.",
                path, exc,
            )
            return cls([])

        if isinstance(data, dict):
            entries = data.get("apps", [])
        elif isinstance(data, list):
            entries = data
        else:
            log.warning("Catalog at %s is not a recognised shape.", path)
            return cls([])

        apps: List[KnownApp] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            aliases = [
                a.strip() for a in (entry.get("aliases") or [])
                if isinstance(a, str) and a.strip()
            ]
            packages: List[PackageVariant] = []
            # New form: "packages": [{"package": "...", "version_prefixes": [...]}, ...]
            raw_packages = entry.get("packages")
            if isinstance(raw_packages, list):
                for v in raw_packages:
                    if isinstance(v, str) and v.strip():
                        packages.append(PackageVariant(package=v.strip()))
                    elif isinstance(v, dict):
                        pkg = (v.get("package") or "").strip()
                        if pkg:
                            prefixes = [
                                str(p).strip()
                                for p in (v.get("version_prefixes") or [])
                                if str(p).strip()
                            ]
                            packages.append(PackageVariant(
                                package=pkg, version_prefixes=prefixes,
                            ))
            # Legacy single-package form.
            elif isinstance(entry.get("package"), str) \
                    and entry["package"].strip():
                packages.append(PackageVariant(package=entry["package"].strip()))

            apps.append(KnownApp(
                name=name, aliases=aliases, packages=packages,
                requires_context=bool(entry.get("requires_context")),
            ))
        log.info("Loaded %d known apps from %s", len(apps), path)
        return cls(apps)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    @property
    def apps(self) -> List[KnownApp]:
        return list(self._apps)

    def find_by_package(self, package: str) -> Optional[KnownApp]:
        """Return the catalog entry for ``package``.

        Exact match first; on a miss, walk back from the right and
        retry against any dotted prefix. This catches sub-variants
        the catalog hasn't explicitly registered yet:
        ``com.forvia.zoomapp.rsedemo`` resolves to the ``Zoom`` entry
        via its registered parent ``com.forvia.zoomapp``.
        """
        if not package:
            return None
        key = package.strip().lower()
        exact = self._by_package.get(key)
        if exact is not None:
            return exact
        parts = key.split(".")
        for cut in range(len(parts) - 1, 1, -1):
            prefix = ".".join(parts[:cut])
            hit = self._by_package.get(prefix)
            if hit is not None:
                return hit
        return None

    def find_by_name(self, name: str) -> Optional[KnownApp]:
        if not name:
            return None
        return self._by_alias_lower.get(name.strip().lower())

    def find_in_text(self, text: str) -> List[CatalogMatch]:
        if not text:
            return []
        results: List[CatalogMatch] = []
        claimed: List = []

        def _is_claimed(start: int, length: int) -> bool:
            for c_start, c_len in claimed:
                if start < c_start + c_len and c_start < start + length:
                    return True
            return False

        for alias in self._aliases_by_length:
            app = self._by_alias_lower[alias]
            pattern = (
                r"(?<![A-Za-z0-9_])"
                + re.escape(alias)
                + r"(?![A-Za-z0-9_])"
            )
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                start = m.start()
                length = m.end() - m.start()
                if _is_claimed(start, length):
                    continue
                # For entries flagged ``requires_context``, refuse to
                # accept a bare alias mention — it must appear next to
                # a context cue ("app", "application", "game", "the X",
                # "X app", etc.). Otherwise common English words like
                # "audible" become false-positive Audible matches.
                if app.requires_context and not _has_context_cue(
                    text, start, length,
                ):
                    continue
                claimed.append((start, length))
                window_end = min(
                    len(text),
                    start + length + _NEAR_VERSION_WINDOW,
                )
                window_start = max(0, start - _NEAR_VERSION_WINDOW)
                after_text = text[start + length:window_end]
                vm = _NEAR_VERSION_RE.search(after_text)
                if not vm:
                    before_text = text[window_start:start]
                    vm = _NEAR_VERSION_RE.search(before_text)
                results.append(CatalogMatch(
                    app=app,
                    matched_alias=text[start:start + length],
                    position=start,
                    nearby_version=vm.group(1) if vm else None,
                ))
        results.sort(key=lambda r: r.position)
        return results
