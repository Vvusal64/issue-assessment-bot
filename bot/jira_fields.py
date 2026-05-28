"""Read structured app metadata from a Jira issue's custom fields."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .jira_client import StructuredFields


log = logging.getLogger("bot.jira_fields")


KNOWN_FIELD_NAMES = {
    "app_name": [
        "App Name",
        "Application Name",
        "App",
    ],
    "package_name": [
        "Package Name",
        "Package",
        "Application ID",
        "Application Id",
        "Package ID",
        "Package Id",
    ],
    "version_names": [
        "Version Name",
        "App Version",
        "Application Version",
        "Version",
    ],
    "app_fix_versions": [
        "App Fix Version",
        "Fix Version",
        "Fix Versions",
        "Affects Version",
        "Affects Versions",
    ],
    "version_code": [
        "Version Code",
        "App Version Code",
        "Application Version Code",
        "VersionCode",
        "versionCode",
    ],
    "environment_name": [
        "Environment Name",
        "Test Environment",
        "Test Environment (List)",
        "Environment",
        "Env",
    ],
}


class StructuredFieldReader:
    """Discover Jira custom field IDs and read structured fields per issue."""

    def __init__(self, jira_client) -> None:
        self._client = jira_client
        self._field_id_by_key: Optional[Dict[str, str]] = None

    @property
    def field_id_by_key(self) -> Dict[str, str]:
        self._ensure_loaded()
        return dict(self._field_id_by_key or {})

    def _ensure_loaded(self) -> None:
        if self._field_id_by_key is not None:
            return

        try:
            all_fields = self._client.list_fields()
        except Exception as exc:
            log.warning(
                "Could not list Jira fields: %s — structured-field reading is disabled.",
                exc,
            )
            self._field_id_by_key = {}
            return

        by_name: Dict[str, str] = {}

        for f in all_fields:
            name = (f.get("name") or "").strip()
            field_id = f.get("id") or ""

            if name and field_id:
                by_name.setdefault(name.lower(), field_id)

        resolved: Dict[str, str] = {}

        for logical_key, candidates in KNOWN_FIELD_NAMES.items():
            for cand in candidates:
                fid = by_name.get(cand.lower())
                if fid:
                    resolved[logical_key] = fid
                    log.info(
                        "Jira field resolved: %s -> %s (%s)",
                        logical_key,
                        fid,
                        cand,
                    )
                    break
            else:
                log.debug("No Jira custom field resolved for %s", logical_key)

        self._field_id_by_key = resolved

    def read(self, fields: Dict[str, Any]) -> StructuredFields:
        self._ensure_loaded()

        idmap = self._field_id_by_key or {}
        out = StructuredFields()

        if not fields or not idmap:
            log.warning("[STRUCTURED READ] no fields/idmap available | idmap=%s", idmap)
            return out

        def _value_for(logical: str) -> Any:
            field_id = idmap.get(logical)
            if not field_id:
                return None
            return fields.get(field_id)

        raw_app_name = _value_for("app_name")
        raw_package = _value_for("package_name")
        raw_versions = _value_for("version_names")
        raw_fix_versions = _value_for("app_fix_versions")
        raw_version_code = _value_for("version_code")
        raw_environment = _value_for("environment_name")

        out.app_name = coerce_string(raw_app_name)
        out.package_name = coerce_package(raw_package)
        out.version_names = coerce_versions(raw_versions)
        out.app_fix_versions = coerce_versions(raw_fix_versions)
        out.version_code = coerce_string(raw_version_code)
        out.environment_name = coerce_string(raw_environment)

        log.warning(
            "[STRUCTURED READ RESULT] app_name=%r package=%r versions=%r "
            "fix_versions=%r version_code=%r environment=%r",
            out.app_name,
            out.package_name,
            out.version_names,
            out.app_fix_versions,
            out.version_code,
            out.environment_name,
        )

        return out


_VERSION_VALUE_RE = re.compile(
    r"\d+(?:\.\d+){0,3}(?:[-_+.][A-Za-z0-9]+)?"
)

_PACKAGE_VALUE_RE = re.compile(
    r"\b([a-z][a-z0-9_]+(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)\b"
)

_VERSION_SPLIT_RE = re.compile(r"[,;/]| and | & |\n|\r|\s\|\s|\s\\\s")


def coerce_string(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        s = value.strip()
        return s or None

    if isinstance(value, dict):
        return coerce_string(
            value.get("value")
            or value.get("name")
            or value.get("displayName")
            or value.get("label")
        )

    if isinstance(value, list):
        for item in value:
            s = coerce_string(item)
            if s:
                return s

    return None


def coerce_package(value: Any) -> Optional[str]:
    s = coerce_string(value)
    if not s:
        return None

    m = _PACKAGE_VALUE_RE.search(s)
    return m.group(1) if m else None


def coerce_versions(value: Any) -> List[str]:
    out: List[str] = []
    seen: set = set()

    def _emit(piece: str) -> None:
        piece = piece.strip().lstrip("vV")

        if not piece:
            return

        if _VERSION_VALUE_RE.fullmatch(piece):
            version = piece
        else:
            m = _VERSION_VALUE_RE.search(piece)
            if not m:
                return
            version = m.group(0)

        key = version.lower()

        if key not in seen:
            seen.add(key)
            out.append(version)

    def _walk(v: Any) -> None:
        if v is None:
            return

        if isinstance(v, str):
            for part in _VERSION_SPLIT_RE.split(v):
                _emit(part)
            return

        if isinstance(v, dict):
            _walk(
                v.get("name")
                or v.get("value")
                or v.get("displayName")
                or v.get("label")
            )
            return

        if isinstance(v, list):
            for item in v:
                _walk(item)

    _walk(value)
    return out