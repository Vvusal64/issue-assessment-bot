"""Admin Portal resolver — verifies app metadata against the Appning
back-office.

This is the long-term source of truth: an app's package, name, and
currently published versions all live in the Admin Portal. The
extractor uses this resolver as the final verification step — once
the bot has decided what the metadata probably is, it asks the portal
"is this what you have on file?" and accepts the portal's values when
they're available.

The resolver is **optional**. When ``ADMIN_PORTAL_URL`` /
``ADMIN_PORTAL_TOKEN`` are not configured, the chain in
``main.py`` simply doesn't include this resolver and the bot falls
back to local-mapping behaviour.

Endpoints (assumed REST conventions — adjust to fit the real API
when it's ready):

* ``GET  {base}/apps?package={pkg}``    → ``{name, package, versions}``
* ``GET  {base}/apps?name={name}``      → ``[{name, package, versions}, …]``
* ``GET  {base}/apps/{pkg}/versions``   → ``[{version: "..."}, …]``

Authentication is a Bearer token by default (override
``_auth_header()`` for other schemes).

The class also implements a ``resolve_full(...)`` method so the
extractor's existing post-merge verification step (which uses
``getattr(resolver, "resolve_full", None)``) works without any
additional wiring.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .app_name_resolver import AppNameResolver
from .utils import retry


log = logging.getLogger("bot.admin_portal_resolver")


_DEFAULT_SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__), "admin_portal_snapshot.json",
)


@dataclass
class AdminPortalApp:
    """Result shape returned by the Admin Portal."""
    name: Optional[str] = None
    package: Optional[str] = None
    versions: List[str] = field(default_factory=list)


class AdminPortalResolver(AppNameResolver):
    """Verify and look up app metadata against the Admin Portal."""

    provides_admin_verification = True

    def __init__(
        self,
        base_url: str,
        api_token: str,
        *,
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = api_token
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({
            "User-Agent": "jira-assessment-bot/1.0",
            "Accept": "application/json",
            **self._auth_header(),
        })
        # Tiny per-process cache so we don't hammer the portal when the
        # same package shows up across many issues in one run.
        self._cache: Dict[str, AdminPortalApp] = {}

    # ------------------------------------------------------------------
    # AppNameResolver interface
    # ------------------------------------------------------------------

    def resolve(self, package_name: str) -> Optional[str]:
        if not package_name:
            return None
        record = self._lookup_by_package(package_name)
        return record.name if record else None

    def resolve_versions(self, package_name: str) -> List[str]:
        if not package_name:
            return []
        record = self._lookup_by_package(package_name)
        return list(record.versions) if record else []

    def resolve_package_by_name(self, app_name: str) -> Optional[str]:
        """Reverse lookup — used when the bot detected an app name in
        free text (e.g. "Zoom") but no package was visible. Returns the
        canonical package coordinate from the portal, or ``None``.
        """
        if not app_name:
            return None
        records = self._lookup_by_name(app_name)
        if not records:
            return None
        # Prefer an exact case-insensitive name match; fall back to the
        # first hit.
        for r in records:
            if r.name and r.name.strip().lower() == app_name.strip().lower():
                return r.package or None
        return records[0].package or None

    # ------------------------------------------------------------------
    # Used by RegexExtractor's post-merge verification step
    # ------------------------------------------------------------------

    def resolve_full(
        self, *,
        package: Optional[str] = None,
        app_name: Optional[str] = None,
        package_name: Optional[str] = None,  # alias accepted by extractor
    ) -> Optional[Dict[str, Any]]:
        """Return ``{app_name, package_name, versions}`` from the
        portal, or ``None`` if nothing matches.

        Tries package lookup first (most specific), then falls back to
        name lookup. Caller is the extractor's verification step which
        accepts either ``package=`` or ``package_name=``.
        """
        pkg = package or package_name
        record: Optional[AdminPortalApp] = None
        if pkg:
            record = self._lookup_by_package(pkg)
        if record is None and app_name:
            records = self._lookup_by_name(app_name)
            record = records[0] if records else None
        if record is None:
            return None
        return {
            "app_name": record.name,
            "package_name": record.package,
            "versions": list(record.versions),
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _auth_header(self) -> Dict[str, str]:
        """Override in subclasses to support different auth schemes."""
        return {"Authorization": f"Bearer {self._token}"}

    @retry(logger=log)
    def _get(self, path: str, *, params: Optional[Dict[str, Any]] = None,
             ) -> requests.Response:
        url = f"{self._base_url}{path}"
        return self._session.get(url, params=params, timeout=self._timeout)

    def _lookup_by_package(self, package: str) -> Optional[AdminPortalApp]:
        key = f"pkg:{package.lower()}"
        if key in self._cache:
            return self._cache[key]
        try:
            resp = self._get("/apps", params={"package": package})
        except requests.RequestException as exc:
            log.warning("Admin Portal package lookup failed (%s): %s",
                        package, exc)
            return None
        if resp.status_code == 404:
            self._cache[key] = AdminPortalApp()
            return None
        if not resp.ok:
            log.warning("Admin Portal returned HTTP %d for package %s",
                        resp.status_code, package)
            return None
        record = self._parse_record(resp.json())
        if record is not None:
            self._cache[key] = record
        return record

    def _lookup_by_name(self, name: str) -> List[AdminPortalApp]:
        key = f"name:{name.strip().lower()}"
        cached = self._cache.get(key)
        if cached is not None:
            return [cached] if cached.name else []
        try:
            resp = self._get("/apps", params={"name": name})
        except requests.RequestException as exc:
            log.warning("Admin Portal name lookup failed (%s): %s",
                        name, exc)
            return []
        if resp.status_code == 404:
            return []
        if not resp.ok:
            log.warning("Admin Portal returned HTTP %d for name %s",
                        resp.status_code, name)
            return []
        data = resp.json()
        items = data if isinstance(data, list) else [data]
        records = [
            r for r in (self._parse_record(item) for item in items) if r
        ]
        if records:
            # Cache the first hit by name for cheap reuse.
            self._cache[key] = records[0]
        return records

    @staticmethod
    def _parse_record(data: Any) -> Optional[AdminPortalApp]:
        """Normalise an Admin Portal payload into an
        :class:`AdminPortalApp`. Tolerates several common naming
        conventions (``app_name``/``name``, ``package``/``package_name``
        /``application_id``, ``versions``/``version_names``).
        """
        if not isinstance(data, dict):
            return None
        name = (
            data.get("name")
            or data.get("app_name")
            or data.get("application_name")
        )
        package = (
            data.get("package")
            or data.get("package_name")
            or data.get("application_id")
            or data.get("applicationId")
        )
        raw_versions: Any = (
            data.get("versions")
            or data.get("version_names")
            or data.get("app_versions")
            or []
        )
        versions: List[str] = []
        if isinstance(raw_versions, str):
            versions = [v.strip() for v in raw_versions.split(",") if v.strip()]
        elif isinstance(raw_versions, list):
            for v in raw_versions:
                if isinstance(v, str) and v.strip():
                    versions.append(v.strip())
                elif isinstance(v, dict):
                    val = v.get("name") or v.get("version") or v.get("value")
                    if isinstance(val, str) and val.strip():
                        versions.append(val.strip())
        if not (name or package or versions):
            return None
        return AdminPortalApp(
            name=(name.strip() if isinstance(name, str) else None) or None,
            package=(package.strip() if isinstance(package, str) else None) or None,
            versions=versions,
        )


class LocalAdminPortalSnapshotResolver(AppNameResolver):
    """Authoritative-but-offline stand-in for the live Admin Portal.

    Reads ``bot/admin_portal_snapshot.json`` — a curated file that
    mirrors what the user sees on the Appning Admin Portal's
    Certification page. Used when the live ``ADMIN_PORTAL_URL`` /
    ``ADMIN_PORTAL_TOKEN`` credentials are not configured.

    *Why offline?* The bot must never publish "high confidence" results
    that haven't been cross-checked against the portal. Real-world
    Jira tickets routinely contain misleading evidence — a publisher
    name dressed up as an app name (``Streamingmedia`` vs.
    ``Radio Format``), a developer's internal codename, a system
    package mistakenly logged as the app under test. Until the live
    API is wired up, this snapshot lets the bot verify against a
    trusted record without making HTTP calls.

    Sets ``provides_admin_verification = True`` so the extractor's
    strict gate trusts a ``None`` result as "this package isn't on
    file" (a ``no_match`` that downgrades confidence) rather than
    "this resolver doesn't do verification" (legacy ``not_checked``).

    Snapshot file shape (see ``admin_portal_snapshot.json``)::

        {
          "apps": [
            {
              "name": "Radio Format",
              "package": "it.streamingmedia.radioformat",
              "publisher": "LUIGI PETRUCCIO",
              "versions": ["1.2.0", "1.0.1", "1.0.0"]
            },
            ...
          ]
        }

    Lookups are case-insensitive on package coordinates and on names.
    """

    provides_admin_verification = True

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or _DEFAULT_SNAPSHOT_PATH
        self._by_package: Dict[str, AdminPortalApp] = {}
        self._by_name: Dict[str, AdminPortalApp] = {}
        self._load()

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            log.warning(
                "Admin Portal snapshot at %s could not be loaded: %s — "
                "snapshot is empty, the bot will not be able to "
                "verify any extraction.", self._path, exc,
            )
            return

        entries = (
            data.get("apps", []) if isinstance(data, dict)
            else data if isinstance(data, list)
            else []
        )

        loaded = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("name") or "").strip()
            package = (entry.get("package") or "").strip()
            if not (name and package):
                continue
            raw_versions = entry.get("versions") or []
            versions: List[str] = []
            if isinstance(raw_versions, list):
                versions = [
                    str(v).strip() for v in raw_versions
                    if isinstance(v, (str, int, float)) and str(v).strip()
                ]
            record = AdminPortalApp(
                name=name, package=package, versions=versions,
            )
            self._by_package.setdefault(package.lower(), record)
            # Multiple packages can share a name (e.g. Zoom for Cars
            # has push_notifications + rsedemo variants). Keep the
            # FIRST seen — that's the canonical one in the file.
            self._by_name.setdefault(name.lower(), record)
            loaded += 1

        log.info(
            "Admin Portal snapshot loaded: %d entries from %s",
            loaded, self._path,
        )

    # ------------------------------------------------------------------
    # AppNameResolver interface
    # ------------------------------------------------------------------

    def resolve(self, package_name: str) -> Optional[str]:
        if not package_name:
            return None
        record = self._by_package.get(package_name.strip().lower())
        return record.name if record else None

    def resolve_versions(self, package_name: str) -> List[str]:
        if not package_name:
            return []
        record = self._by_package.get(package_name.strip().lower())
        return list(record.versions) if record else []

    def resolve_package_by_name(self, app_name: str) -> Optional[str]:
        if not app_name:
            return None
        record = self._by_name.get(app_name.strip().lower())
        return record.package if record else None

    def resolve_full(
        self,
        package: Optional[str] = None,
        app_name: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return ``{app_name, package_name, versions}`` from the
        snapshot, or ``None`` if the package isn't known.

        Package lookup wins over name lookup — a package coordinate
        is unambiguous; a name can collide across variants.
        """
        pkg = package or package_name
        record: Optional[AdminPortalApp] = None
        if pkg:
            record = self._by_package.get(pkg.strip().lower())
        if record is None and app_name:
            record = self._by_name.get(app_name.strip().lower())
        if record is None:
            return None
        return {
            "app_name": record.name,
            "package_name": record.package,
            "versions": list(record.versions),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_package)

    def has_package(self, package: str) -> bool:
        """Membership probe used by the extractor's strict gate."""
        if not package:
            return False
        return package.strip().lower() in self._by_package
