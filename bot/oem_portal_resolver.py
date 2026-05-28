"""OEM-specific portal resolvers (BMW, Mercedes, Changan, …).

The Appning Admin Portal holds the canonical app catalogue, but each
OEM also runs its own portal where the same app might appear under a
different package coordinate or with a different set of approved
versions (e.g. ``com.forvia.zoomapp.rsedemo`` vs
``com.forvia.zoomapp`` for BMW vs Mercedes). This module gives us a
single interface for those portals — chained AFTER the Appning Admin
Portal in the main resolver chain — so an issue tagged ``BMW`` can be
double-checked against the BMW portal before the bot sends a Slack
notification.

When credentials aren't configured the resolvers are simply not
included in the chain, and the bot's behaviour is unchanged.

Adding a new OEM is one line in ``OEM_PORTALS`` plus two env vars:
``<OEM>_PORTAL_URL`` / ``<OEM>_PORTAL_TOKEN``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .admin_portal_resolver import AdminPortalResolver


log = logging.getLogger("bot.oem_portal_resolver")


# OEM short-codes we know about. Adding a new OEM is a one-line entry
# here — the wiring is otherwise generic.
OEM_PORTALS: Dict[str, str] = {
    "BMW": "BMW",
    "MERCEDES": "MERCEDES",
    "CHANGAN": "CHANGAN",
    "GEELY": "GEELY",
    "SAIC": "SAIC",
    "VINFAST": "VINFAST",
}


@dataclass
class OEMPortalConfig:
    oem: str
    base_url: str
    api_token: str


class OEMPortalResolver(AdminPortalResolver):
    """A thin specialisation of :class:`AdminPortalResolver` that
    tags log lines with the OEM short-code and lets the rest of the
    chain keep using the same Bearer-token HTTP pattern.

    The actual endpoint paths and payload shapes vary per OEM; when
    that becomes a real concern, override ``_lookup_by_package`` /
    ``_lookup_by_name`` in an OEM-specific subclass below.
    """

    def __init__(self, oem: str, base_url: str, api_token: str,
                 **kwargs) -> None:
        self._oem = oem.upper()
        super().__init__(base_url=base_url, api_token=api_token, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<OEMPortalResolver oem={self._oem!r} base_url={self._base_url!r}>"


def load_oem_resolvers_from_env() -> List[OEMPortalResolver]:
    """Return one :class:`OEMPortalResolver` for each OEM that has
    both ``<OEM>_PORTAL_URL`` and ``<OEM>_PORTAL_TOKEN`` set.

    Example::

        export BMW_PORTAL_URL=https://bmw.appning.example/api/v1
        export BMW_PORTAL_TOKEN=…
        export MERCEDES_PORTAL_URL=https://mercedes.appning.example/api/v1
        export MERCEDES_PORTAL_TOKEN=…

    The orchestrator chains the returned resolvers after the main
    AdminPortalResolver in the resolver chain.
    """
    out: List[OEMPortalResolver] = []
    for oem in OEM_PORTALS:
        url = (os.environ.get(f"{oem}_PORTAL_URL") or "").strip()
        token = (os.environ.get(f"{oem}_PORTAL_TOKEN") or "").strip()
        if not url or not token:
            continue
        try:
            resolver = OEMPortalResolver(
                oem=oem, base_url=url, api_token=token,
            )
        except Exception as exc:  # never block startup on this
            log.warning("OEM portal %s init failed: %s", oem, exc)
            continue
        log.info("OEM portal resolver enabled: %s -> %s", oem, url)
        out.append(resolver)
    return out


def detect_oem_from_issue_key(issue_key: Optional[str]) -> Optional[str]:
    """Best-effort OEM detection from a Jira issue key prefix.

    BMW-3455 → 'BMW'; VF-578 → None (we don't have a 'VF' portal
    configured by default — add to ``OEM_PORTALS`` when needed).
    Returns ``None`` when the prefix isn't recognised; this is just
    a debugging aid for log lines.
    """
    if not issue_key or "-" not in issue_key:
        return None
    prefix = issue_key.split("-", 1)[0].upper()
    if prefix in OEM_PORTALS:
        return prefix
    return None
