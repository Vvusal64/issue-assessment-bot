"""Tests for the Jira structured-field reader.

Covers ID resolution by display name, value coercion (strings, lists,
Jira version objects), multi-version splitting, and graceful failure
when Jira is unreachable or the JSON is junk.
"""

from __future__ import annotations

import unittest

from bot.jira_client import StructuredFields
from bot.jira_fields import (
    StructuredFieldReader,
    coerce_package,
    coerce_string,
    coerce_versions,
)


class _FakeJiraClient:
    """In-memory stand-in for :class:`JiraClient`."""

    def __init__(self, fields):
        self._fields = fields
        self.list_calls = 0

    def list_fields(self):
        self.list_calls += 1
        return list(self._fields)


class _BoomClient:
    def list_fields(self):
        raise RuntimeError("jira down")


# ----------------------------------------------------------------------
# Coercion helpers
# ----------------------------------------------------------------------

class CoerceStringTests(unittest.TestCase):

    def test_string_passthrough(self):
        self.assertEqual(coerce_string("Bild TV"), "Bild TV")

    def test_strips_whitespace(self):
        self.assertEqual(coerce_string("  Deezer\n"), "Deezer")

    def test_select_list_object(self):
        self.assertEqual(coerce_string({"value": "Spotify"}), "Spotify")

    def test_version_object(self):
        self.assertEqual(coerce_string({"name": "1.2.3"}), "1.2.3")

    def test_list_first_truthy(self):
        self.assertEqual(coerce_string(["", "  ", "Waze"]), "Waze")


class CoercePackageTests(unittest.TestCase):

    def test_plain_package(self):
        self.assertEqual(
            coerce_package("com.mekmedia.bild.auto"),
            "com.mekmedia.bild.auto",
        )

    def test_strips_markdown_link(self):
        self.assertEqual(
            coerce_package(
                "[com.mekmedia.bild.auto](http://com.mekmedia.bild.auto)"
            ),
            "com.mekmedia.bild.auto",
        )

    def test_returns_none_for_non_package_text(self):
        self.assertIsNone(coerce_package("just a sentence"))


class CoerceVersionsTests(unittest.TestCase):

    def test_single_version(self):
        self.assertEqual(coerce_versions("3.0.0.5"), ["3.0.0.5"])

    def test_comma_separated(self):
        self.assertEqual(
            coerce_versions("3.0.0.5, 3.1.13"), ["3.0.0.5", "3.1.13"],
        )

    def test_and_separator(self):
        self.assertEqual(
            coerce_versions("3.0.0.5 and 3.1.13"), ["3.0.0.5", "3.1.13"],
        )

    def test_semicolon_separator(self):
        self.assertEqual(
            coerce_versions("3.0.0.5; 3.1.13"), ["3.0.0.5", "3.1.13"],
        )

    def test_jira_version_objects_list(self):
        # Jira "Affects Version" / "Fix Version" type fields come back
        # as a list of {"name": "..."} objects.
        self.assertEqual(
            coerce_versions([{"name": "3.0.0.5"}, {"name": "3.1.13"}]),
            ["3.0.0.5", "3.1.13"],
        )

    def test_de_duplicates(self):
        self.assertEqual(
            coerce_versions("3.0.0.5, 3.0.0.5, 3.1.13"),
            ["3.0.0.5", "3.1.13"],
        )

    def test_strips_v_prefix(self):
        self.assertEqual(coerce_versions("v3.0.0.1"), ["3.0.0.1"])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(coerce_versions(None), [])
        self.assertEqual(coerce_versions(""), [])


# ----------------------------------------------------------------------
# StructuredFieldReader
# ----------------------------------------------------------------------

class StructuredFieldReaderTests(unittest.TestCase):

    def _bmw_3362_setup(self):
        # Pretend the Jira instance has these custom field IDs.
        client = _FakeJiraClient([
            {"id": "summary", "name": "Summary"},
            {"id": "customfield_10100", "name": "App Name"},
            {"id": "customfield_10101", "name": "Package Name"},
            {"id": "customfield_10102", "name": "Version Name"},
            {"id": "customfield_10103", "name": "App Fix Version"},
        ])
        return client, StructuredFieldReader(client)

    def test_resolves_known_field_ids_lazily(self):
        client, reader = self._bmw_3362_setup()
        self.assertEqual(client.list_calls, 0)
        ids = reader.field_id_by_key
        self.assertEqual(client.list_calls, 1)
        self.assertEqual(ids["app_name"], "customfield_10100")
        self.assertEqual(ids["package_name"], "customfield_10101")
        # The catalog uses plural keys (``version_names`` /
        # ``app_fix_versions``) since these fields naturally hold
        # multiple values.
        self.assertEqual(ids["version_names"], "customfield_10102")
        # Subsequent reads use the cache.
        _ = reader.field_id_by_key
        self.assertEqual(client.list_calls, 1)

    def test_reads_bmw_3362_fields(self):
        _, reader = self._bmw_3362_setup()
        fields = {
            "summary": "[IDC23] something broke",
            "customfield_10100": "Bild TV",
            "customfield_10101": (
                "[com.mekmedia.bild.auto]"
                "(http://com.mekmedia.bild.auto)"
            ),
            "customfield_10102": "3.0.0.5 and 3.1.13",
        }
        result = reader.read(fields)
        self.assertEqual(result.app_name, "Bild TV")
        self.assertEqual(result.package_name, "com.mekmedia.bild.auto")
        self.assertEqual(result.version_names, ["3.0.0.5", "3.1.13"])
        self.assertEqual(result.app_fix_versions, [])
        self.assertTrue(result.has_any())

    def test_aliases_match(self):
        # Some Jira instances label these fields differently. Make sure
        # the alias list catches a couple of common variants.
        client = _FakeJiraClient([
            {"id": "customfield_1", "name": "Application Name"},
            {"id": "customfield_2", "name": "Application ID"},
            {"id": "customfield_3", "name": "App Version"},
        ])
        reader = StructuredFieldReader(client)
        fields = {
            "customfield_1": "Deezer",
            "customfield_2": "deezer.android.app",
            "customfield_3": "3.0.0.1",
        }
        result = reader.read(fields)
        self.assertEqual(result.app_name, "Deezer")
        self.assertEqual(result.package_name, "deezer.android.app")
        self.assertEqual(result.version_names, ["3.0.0.1"])

    def test_returns_empty_when_jira_is_down(self):
        # list_fields() raising must not break the bot — we just don't
        # get any structured-field signal for this run.
        reader = StructuredFieldReader(_BoomClient())
        result = reader.read({"customfield_10100": "Bild TV"})
        self.assertEqual(result, StructuredFields())
        self.assertFalse(result.has_any())

    def test_returns_empty_when_fields_are_absent(self):
        _, reader = self._bmw_3362_setup()
        # No structured field values at all.
        result = reader.read({"summary": "x"})
        self.assertFalse(result.has_any())


if __name__ == "__main__":
    unittest.main()
