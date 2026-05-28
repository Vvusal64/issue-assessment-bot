"""Tests for related-ticket JQL construction.

Covers the priority order (package > app_name > none), the deliberate
absence of version-only searches, and JQL string escaping.
"""

from __future__ import annotations

import unittest

from bot.jira_client import build_related_open_jql


class BuildRelatedOpenJqlTests(unittest.TestCase):

    def test_package_takes_priority_over_app_name(self) -> None:
        jql = build_related_open_jql(
            exclude_key="BMW-3370",
            package_name="deezer.android.app",
            app_name="Deezer",
        )
        # Package is preferred — app_name must NOT appear in the JQL.
        self.assertIn('text ~ "deezer.android.app"', jql)
        self.assertNotIn("Deezer", jql)
        # Issue type, resolution and exclude_key all enforced.
        self.assertIn('type = "OEM App Bug"', jql)
        self.assertIn("resolution = Unresolved", jql)
        self.assertIn('key != "BMW-3370"', jql)
        self.assertIn("ORDER BY updated DESC", jql)

    def test_app_name_alone_no_longer_produces_a_query(self) -> None:
        # New policy: related-tickets are strictly same-package. If we
        # don't have a package, we don't search at all — an app_name-
        # only fallback would surface tickets about a different
        # variant of the same family (Stellantis Zoom vs Forvia Zoom
        # for Cars), which is exactly what we're avoiding.
        self.assertIsNone(build_related_open_jql(
            exclude_key="VF-533",
            package_name=None,
            app_name="Blanco",
        ))

    def test_returns_none_when_neither_provided(self) -> None:
        self.assertIsNone(build_related_open_jql(
            exclude_key="X-1", package_name=None, app_name=None,
        ))
        self.assertIsNone(build_related_open_jql(
            exclude_key="X-1", package_name="", app_name="",
        ))

    def test_version_only_is_not_supported(self) -> None:
        # The signature deliberately has no app_version parameter.
        # Version-only callers should pass nothing and get None.
        self.assertIsNone(build_related_open_jql(
            exclude_key="X-1", package_name=None, app_name=None,
        ))

    def test_uses_structured_package_after_regex_disagrees(self) -> None:
        # Caller is expected to pass the *verified* (post-merge) package
        # — i.e. the structured field's value when it disagrees with
        # regex. This test documents that contract: as long as the
        # caller hands us the structured value, the JQL searches for it.
        verified_package = "com.mekmedia.bild.auto"
        regex_only_guess = "com.regex.fake"
        jql = build_related_open_jql(
            exclude_key="BMW-3362",
            package_name=verified_package,
            app_name=None,
        )
        self.assertIn(f'text ~ "{verified_package}"', jql)
        self.assertNotIn(regex_only_guess, jql)

    def test_jql_strings_are_escaped(self) -> None:
        # Hostile-but-plausible inputs must not break the JQL.
        jql = build_related_open_jql(
            exclude_key='X"-1',
            package_name='com.example."weird"',
            app_name=None,
        )
        # Both " characters in inputs become \" in the output.
        self.assertIn(r'\"weird\"', jql)
        self.assertIn(r'X\"-1', jql)


if __name__ == "__main__":
    unittest.main()
