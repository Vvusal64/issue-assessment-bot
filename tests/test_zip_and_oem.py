"""Tests for zip-attachment expansion and OEM portal config loading."""

from __future__ import annotations

import io
import os
import unittest
import zipfile
from unittest import mock

from bot.jira_client import Attachment, JiraClient
from bot.oem_portal_resolver import (
    detect_oem_from_issue_key,
    load_oem_resolvers_from_env,
)


# ---------------------------------------------------------------------
# OEM detection / loader
# ---------------------------------------------------------------------

class OemDetectionTests(unittest.TestCase):
    def test_known_prefix(self) -> None:
        self.assertEqual(detect_oem_from_issue_key("BMW-3455"), "BMW")
        self.assertEqual(detect_oem_from_issue_key("CHANGAN-12"), "CHANGAN")

    def test_unknown_prefix(self) -> None:
        # VF is a real prefix in your Jira but not yet in OEM_PORTALS;
        # the bot must return None rather than guess.
        self.assertIsNone(detect_oem_from_issue_key("VF-578"))
        self.assertIsNone(detect_oem_from_issue_key("NO-DASH"))
        self.assertIsNone(detect_oem_from_issue_key(None))


class OemLoaderTests(unittest.TestCase):
    def test_no_env_returns_empty(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            for var in (
                "BMW_PORTAL_URL", "BMW_PORTAL_TOKEN",
                "MERCEDES_PORTAL_URL", "MERCEDES_PORTAL_TOKEN",
            ):
                os.environ.pop(var, None)
            self.assertEqual(load_oem_resolvers_from_env(), [])

    def test_partial_env_skipped(self) -> None:
        # URL without token (or vice-versa) is treated as "not configured".
        with mock.patch.dict(
            os.environ,
            {"BMW_PORTAL_URL": "https://x.example"},
            clear=False,
        ):
            os.environ.pop("BMW_PORTAL_TOKEN", None)
            self.assertEqual(load_oem_resolvers_from_env(), [])

    def test_full_env_returns_resolver(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "BMW_PORTAL_URL": "https://bmw.example/api/v1",
                "BMW_PORTAL_TOKEN": "secret-token",
            },
            clear=False,
        ):
            resolvers = load_oem_resolvers_from_env()
        # At least the BMW one comes back. (Other OEMs may also be
        # present if the developer's env has them; we don't assert
        # length exactly to keep this robust.)
        names = [r._oem for r in resolvers]
        self.assertIn("BMW", names)


# ---------------------------------------------------------------------
# Zip extraction
# ---------------------------------------------------------------------

class ZipExpansionTests(unittest.TestCase):
    """Drive ``JiraClient.expand_zip_attachment`` against an in-memory
    zip without touching the network."""

    def _build_zip_bytes(self, entries: dict) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in entries.items():
                zf.writestr(name, content)
        return buf.getvalue()

    def _fake_client_with_zip(self, zip_bytes: bytes) -> JiraClient:
        client = JiraClient.__new__(JiraClient)
        client.base_url = "https://example.invalid"
        client.timeout = 30.0
        # Bypass __init__; install a session double whose .get returns
        # a streaming-shaped response with zip_bytes.
        class _Resp:
            ok = True
            status_code = 200
            def raise_for_status(self): pass
            def iter_content(self, chunk_size=64*1024):
                # one chunk
                yield zip_bytes
        class _Session:
            def get(self, url, **kwargs):
                return _Resp()
        client._session = _Session()
        # The retry-wrapped _get just calls self._session.get.
        def _get(url, **kwargs):
            return client._session.get(url, **kwargs)
        client._get = _get
        return client

    def test_expand_finds_text_entries(self) -> None:
        zip_bytes = self._build_zip_bytes({
            "logs/app.log": "packageName=com.forvia.zoomapp.rsedemo\n"
                            "versionName=1.0.8.9\n",
            "media/icon.png": b"\x89PNG\r\n\x1a\n",  # binary, skipped
            "data.json": '{"event": "x"}',
        })
        client = self._fake_client_with_zip(zip_bytes)
        att = Attachment(
            filename="logs.zip",
            mime_type="application/zip",
            size=len(zip_bytes),
            content_url="https://example.invalid/x",
        )
        expanded = client.expand_zip_attachment(att)
        names = [e.filename for e in expanded]
        # The .log and .json entries are text-like; the .png is not.
        self.assertIn("logs.zip:logs/app.log", names)
        self.assertIn("logs.zip:data.json", names)
        self.assertNotIn("logs.zip:media/icon.png", names)
        # Content survives extraction.
        log_text = next(
            e.text for e in expanded if e.filename.endswith("app.log")
        )
        self.assertIn("com.forvia.zoomapp.rsedemo", log_text)
        self.assertIn("1.0.8.9", log_text)

    def test_non_zip_attachment_is_a_noop(self) -> None:
        client = JiraClient.__new__(JiraClient)
        att = Attachment(
            filename="trace.log",
            mime_type="text/plain",
            size=10,
            content_url="x",
        )
        self.assertEqual(client.expand_zip_attachment(att), [])

    def test_corrupt_zip_returns_empty(self) -> None:
        client = self._fake_client_with_zip(b"not a zip")
        att = Attachment(
            filename="broken.zip",
            mime_type="application/zip",
            size=8,
            content_url="x",
        )
        self.assertEqual(client.expand_zip_attachment(att), [])


if __name__ == "__main__":
    unittest.main()
