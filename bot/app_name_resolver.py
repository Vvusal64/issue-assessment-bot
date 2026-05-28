"""Pluggable resolution from a package coordinate to a human-readable
app name.

The default backend is a local JSON file (``bot/package_app_map.json``).
The :class:`AppNameResolver` ABC and :class:`ChainedResolver` make it
trivial to swap in an Admin Portal lookup, a Jira/REST lookup, or any
other backend later — the rest of the bot only depends on
``resolve(package_name) -> Optional[str]``.

Resolvers must never raise on a missing package — they return ``None``.
Transient errors (filesystem I/O, network) are caught here and logged
so a single backend's failure can't break the bot's main flow.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


log = logging.getLogger("bot.app_name_resolver")


class AppNameResolver(ABC):
    """Resolve a package coordinate to app metadata."""

    # Set to True on resolvers that can authoritatively answer
    # "is this package registered in the Admin Portal / OEM portal?"
    # — the extractor uses this to decide whether a None response
    # from ``resolve_full`` means "package not in portal" (a
    # ``no_match`` signal that downgrades confidence) versus
    # "this resolver doesn't claim to verify, ignore me"
    # (legacy ``not_checked`` behaviour).
    provides_admin_verification: bool = False

    @abstractmethod
    def resolve(self, package_name: str) -> Optional[str]:
        """Return the display name for ``package_name`` or ``None``."""

    def resolve_versions(self, package_name: str) -> List[str]:
        """Return known versions for ``package_name``."""
        return []

    def resolve_package_by_name(self, app_name: str) -> Optional[str]:
        """Reverse lookup — given a human-readable app name, return the
        canonical package coordinate. Default returns ``None`` so
        backends without reverse-lookup support stay simple. Admin
        Portal / catalog backends override this.
        """
        return None

    def resolve_full(
        self,
        package: Optional[str] = None,
        app_name: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return full verified metadata if supported by the resolver.

        Expected shape:
        {
            "app_name": "...",
            "package_name": "...",
            "versions": ["..."]
        }

        Default implementation returns None so existing resolvers stay safe.
        """
        return None


class JsonFileAppNameResolver(AppNameResolver):
    """Default resolver, backed by a flat JSON file."""

    DEFAULT_PATH = os.path.join(
        os.path.dirname(__file__), "package_app_map.json",
    )

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or self.DEFAULT_PATH
        self._cache: Optional[Dict[str, str]] = None

    def reload(self) -> None:
        """Drop the cached mapping."""
        self._cache = None

    def _load(self) -> Dict[str, str]:
        if self._cache is not None:
            return self._cache

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            log.debug("No package→app map at %s: %s", self._path, exc)
            self._cache = {}
            return self._cache

        if not isinstance(data, dict):
            log.warning("%s is not a JSON object — ignoring.", self._path)
            self._cache = {}
            return self._cache

        self._cache = {
            str(k).strip(): str(v).strip()
            for k, v in data.items()
            if isinstance(k, str) and isinstance(v, str) and v.strip()
        }
        return self._cache

    def resolve(self, package_name: str) -> Optional[str]:
        if not package_name:
            return None
        return self._load().get(package_name.strip()) or None

    # Note: ``resolve_full`` is intentionally NOT overridden here.
    # The local JSON map only knows package → name; it has no version
    # data and no genuine "verification" — that's the AdminPortalResolver's
    # job. Returning a result from ``resolve_full`` would falsely flip
    # the extractor's ``admin_verified`` flag and inflate confidence.


class ChainedResolver(AppNameResolver):
    """Try resolvers in order; return the first non-empty hit."""

    def __init__(self, *resolvers: AppNameResolver) -> None:
        self._resolvers: tuple = resolvers
        # The chain is portal-capable if ANY member is — we want a
        # single-config "live portal + snapshot + local JSON map"
        # chain to count as portal-capable so the extractor's strict
        # gate fires.
        self.provides_admin_verification = any(
            getattr(r, "provides_admin_verification", False)
            for r in resolvers
        )

    def resolve(self, package_name: str) -> Optional[str]:
        for resolver in self._resolvers:
            try:
                hit = resolver.resolve(package_name)
            except Exception as exc:
                log.warning(
                    "Resolver %s failed: %s — continuing.",
                    type(resolver).__name__,
                    exc,
                )
                continue

            if hit:
                return hit

        return None

    def resolve_versions(self, package_name: str) -> List[str]:
        for resolver in self._resolvers:
            try:
                hit = resolver.resolve_versions(package_name)
            except Exception as exc:
                log.warning(
                    "Resolver %s.resolve_versions failed: %s",
                    type(resolver).__name__,
                    exc,
                )
                continue

            if hit:
                return list(hit)

        return []

    def resolve_package_by_name(self, app_name: str) -> Optional[str]:
        for resolver in self._resolvers:
            try:
                hit = resolver.resolve_package_by_name(app_name)
            except Exception as exc:
                log.warning(
                    "Resolver %s.resolve_package_by_name failed: %s",
                    type(resolver).__name__,
                    exc,
                )
                continue

            if hit:
                return hit

        return None

    def resolve_full(
        self,
        package: Optional[str] = None,
        app_name: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pkg = package or package_name

        for resolver in self._resolvers:
            try:
                hit = resolver.resolve_full(
                    package=pkg,
                    app_name=app_name,
                    package_name=pkg,
                )
            except Exception as exc:
                log.warning(
                    "Resolver %s.resolve_full failed: %s — continuing.",
                    type(resolver).__name__,
                    exc,
                )
                continue

            if hit:
                return hit

        return None


default_resolver: AppNameResolver = JsonFileAppNameResolver()