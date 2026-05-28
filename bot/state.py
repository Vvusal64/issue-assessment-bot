"""Idempotency store backed by a JSON file.

Atomic writes: the new content is written to a sibling temp file and then
``os.replace``'d into place, so a crash mid-write can't leave the state
file half-written.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


log = logging.getLogger("bot.state")


class StateStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_processed(self, issue_key: str) -> bool:
        with self._lock:
            return issue_key in self._data

    def get(self, issue_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._data.get(issue_key)
            return dict(entry) if entry else None

    def mark_processed(
        self, issue_key: str, *,
        extraction: Dict[str, Any],
        slack_message_ts: Optional[str] = None,
        slack_channel: Optional[str] = None,
    ) -> None:
        record = {
            "processed_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "extraction": extraction,
        }
        if slack_message_ts:
            record["slack_message_ts"] = slack_message_ts
        if slack_channel:
            record["slack_channel"] = slack_channel
        with self._lock:
            self._data[issue_key] = record
            self._flush()
        log.info("State updated for %s", issue_key)

    def reset(self, issue_key: str) -> bool:
        with self._lock:
            if issue_key not in self._data:
                return False
            del self._data[issue_key]
            self._flush()
        log.info("State reset for %s", issue_key)
        return True

    def all_keys(self) -> list:
        with self._lock:
            return list(self._data.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                log.warning("State file %s is not a JSON object; resetting",
                            self._path)
                return {}
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read state file %s: %s — starting empty",
                        self._path, exc)
            return {}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)
        os.replace(tmp, self._path)
