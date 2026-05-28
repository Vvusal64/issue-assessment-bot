"""Unit tests for the regex extractor and confidence scoring.

These tests deliberately exercise realistic Jira issue shapes — log
attachments, mixed-language descriptions, multiple package candidates —
to make sure the heuristics don't degrade silently.

Run from the project root:
    python -m unittest tests.test_extractor -v
"""

from __future__ import annotations

import unittest

from bot.extractor import (
    Extraction,
    RegexExtractor,
    _derive_name_from_package,
    _merge,
)
from bot.jira_client import (
    Attachment, Comment, JiraIssue, StructuredFields,
)


def _issue(*, summary: str = "", description: str = "",
           comments=None, attachments=None,
           structured: StructuredFields = None) -> JiraIssue:
    return JiraIssue(
        key="BMW-9999",
        summary=summary,
        description=description,
        status="Open",
        issue_type="OEM App Bug",
        assignee_display_name="Tester",
        assignee_email="tester@example.com",
        assignee_account_id="acc-1",
        created="2026-04-30T08:00:00.000+0000",
        updated="2026-04-30T08:00:00.000+0000",
        comments=list(comments or []),
        attachments=list(attachments or []),
        structured=structured,
    )


class RegexExtractorTests(unittest.TestCase):

    def test_logcat_attachment_yields_high_confidence(self) -> None:
        # The bracket prefix ("[ExampleMusic]") is intentionally NOT
        # used as the app name — only the explicit ``appLabel:`` line
        # in the log carries that signal.
        log_text = (
            "I/ActivityManager: Start proc 12345 for activity\n"
            "  package=com.example.musicapp\n"
            "  versionName=4.7.2\n"
            "  applicationId: com.example.musicapp\n"
            "  appLabel: ExampleMusic\n"
        )
        issue = _issue(
            summary="[ExampleMusic] - Crash on resume",
            description="See attached log",
            attachments=[Attachment(
                filename="logcat.txt",
                mime_type="text/plain",
                size=len(log_text),
                content_url="https://example.invalid/x",
                text=log_text,
            )],
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "com.example.musicapp")
        self.assertEqual(ext.app_version, "4.7.2")
        self.assertEqual(ext.app_name, "ExampleMusic")
        self.assertEqual(ext.confidence, "high")
        self.assertEqual(ext.evidence["package_name"], "attachment:logcat.txt")

    def test_description_only_medium_confidence(self) -> None:
        # NB: bare "Version: 12.3.4" is intentionally NOT extracted
        # under the v2 policy — only labeled forms (versionName /
        # app_version / applicationVersion) and the app-anchored
        # phrases are accepted.
        desc = (
            "App: ExampleNav\n"
            "Package: com.example.nav\n"
            "app_version: 12.3.4\n"
            "Steps to reproduce: open app, tap search."
        )
        issue = _issue(summary="Map tiles fail to load", description=desc)
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "com.example.nav")
        self.assertEqual(ext.app_version, "12.3.4")
        self.assertIn(ext.confidence, ("medium", "high"))

    def test_multiple_packages_prefers_app_over_system(self) -> None:
        desc = (
            "Stack trace:\n"
            "  at android.os.Looper.loop(Looper.java:201)\n"
            "  at com.partner.coolapp.MainActivity.onCreate(...)\n"
            "  at androidx.fragment.app.Fragment.onResume(...)\n"
            "package: com.partner.coolapp\n"
            "versionName=2.0.1"
        )
        issue = _issue(summary="ANR on resume", description=desc)
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "com.partner.coolapp")
        self.assertEqual(ext.app_version, "2.0.1")

    def test_summary_bracket_metadata_is_never_app_name(self) -> None:
        # Environment / project / vehicle / region tags in a leading
        # bracket must NEVER become the app name. (Real known-app names
        # like "[Spotify]" are now legitimately resolved via the
        # catalog — that's the whole point of the catalog. The check
        # here is specifically that ENV/platform tags don't slip
        # through.)
        for tag in ("[BMW]", "[IDC23]", "[IDCevo]", "[CDE-01]",
                    "[VF7]", "[VF8]", "[EU]"):
            ext = RegexExtractor().extract(
                _issue(summary=f"{tag} - something broke")
            )
            self.assertIsNone(
                ext.app_name,
                f"Bracket prefix {tag!r} should never become app_name",
            )

    # ------------------------------------------------------------------
    # IDC23 / phrase-based extraction
    # ------------------------------------------------------------------

    def test_idc23_phrase_extracts_package_not_env_tag(self) -> None:
        # Spec example 1:
        #   [IDC23] JAVA_CRASH crash due to com.smokoko.careatscar3_appning
        # The package is shipped in the seeded package_app_map.json, so
        # app_name comes from the mapping (capped at "medium").
        issue = _issue(
            summary=(
                "[IDC23] JAVA_CRASH crash due to "
                "com.smokoko.careatscar3_appning"
            )
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(
            ext.package_name, "com.smokoko.careatscar3_appning"
        )
        self.assertEqual(ext.app_name, "Car Eats Car 3")
        self.assertEqual(ext.confidence, "medium")
        self.assertTrue(ext.evidence["package_name"].startswith("summary"))
        # The app name comes from either the catalog (now embeds the
        # matched app's name in the label, e.g.
        # ``catalog[Car Eats Car 3]:summary``) or the legacy package
        # mapping. Both are accepted as evidence-of-known-app sources.
        evidence = ext.evidence["app_name"]
        self.assertTrue(
            evidence.startswith("catalog[")
            or evidence.startswith("catalog_by_package:")
            or evidence.startswith("package_app_map:"),
            f"Unexpected evidence label: {evidence!r}",
        )

    def test_phrase_handles_markdown_link_form(self) -> None:
        # Spec example 2:
        #   [IDC23] JAVA_CRASH crash due to [deezer.android.app](http://…)
        # The package is wrapped in a markdown link; extraction should
        # recover the bare coordinate. With the seeded mapping,
        # app_name resolves to "Deezer".
        log_text = "I/X versionName=3.0.0.1\nI/X package=deezer.android.app\n"
        issue = _issue(
            summary=(
                "[IDC23] JAVA_CRASH crash due to "
                "[deezer.android.app](http://deezer.android.app)"
            ),
            attachments=[Attachment(
                filename="crash.log",
                mime_type="text/plain",
                size=len(log_text),
                content_url="https://example.invalid/x",
                text=log_text,
            )],
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "deezer.android.app")
        self.assertEqual(ext.app_version, "3.0.0.1")
        self.assertEqual(ext.app_name, "Deezer")
        self.assertEqual(ext.confidence, "high")

    def test_caused_by_phrase_extracts_package(self) -> None:
        ext = RegexExtractor().extract(
            _issue(summary="[CDE-01] ANR caused by com.partner.coolapp")
        )
        self.assertEqual(ext.package_name, "com.partner.coolapp")
        # Under the v5 rule "package known → app name known", smart
        # derive produces a name from the package coordinate.
        self.assertIsNotNone(ext.app_name)
        # Never the literal coordinate.
        self.assertNotEqual(ext.app_name, "com.partner.coolapp")

    def test_in_package_phrase_extracts_package(self) -> None:
        ext = RegexExtractor().extract(
            _issue(summary="[VF8] hang in package com.example.maps")
        )
        self.assertEqual(ext.package_name, "com.example.maps")
        # Same rule — package known means a derived name follows.
        self.assertIsNotNone(ext.app_name)
        self.assertNotEqual(ext.app_name, "com.example.maps")

    # ------------------------------------------------------------------
    # Explicit App: / Application: labels
    # ------------------------------------------------------------------

    def test_application_label_in_description_yields_app_name(self) -> None:
        issue = _issue(
            summary="[IDC23] crash on launch",
            description=(
                "Application: Deezer\n"
                "Steps to reproduce: open the app and press play."
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Deezer")

    def test_env_tag_after_label_is_rejected(self) -> None:
        # Even if a careless ticket does "App: IDC23", the guardrail
        # must keep that out of app_name.
        issue = _issue(
            summary="crash",
            description="App: IDC23\nApp: IDCevo\nApp: VF7",
        )
        ext = RegexExtractor().extract(issue)
        self.assertIsNone(ext.app_name)

    def test_app_label_with_package_value_is_not_app_name(self) -> None:
        # "App: com.x.y" reads as a package, not a human-readable
        # name — so the literal coordinate must NEVER end up in
        # app_name. (The bot may still produce a derived name from
        # the recognised package coordinate, just not the coordinate
        # itself.)
        issue = _issue(
            summary="crash",
            description="App: com.example.thing",
        )
        ext = RegexExtractor().extract(issue)
        self.assertNotEqual(ext.app_name, "com.example.thing")
        self.assertEqual(ext.package_name, "com.example.thing")

    # ------------------------------------------------------------------
    # v2: Platform / SW / OS / I-Step / vehicle versions are NEVER
    # extracted as app_version. These are the values the user listed
    # explicitly as forbidden — assert each is silently ignored.
    # ------------------------------------------------------------------

    def test_unity_version_not_extracted_as_app_version(self) -> None:
        issue = _issue(
            summary="[IDC23] crash",
            description=(
                "Unity Version: 2022.3.10\n"
                "No other version data available."
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertIsNone(ext.app_version)

    def test_sw_and_software_version_ignored(self) -> None:
        issue = _issue(
            summary="[IDC23] crash",
            description=(
                "SW Version: 03.62.20\n"
                "Software Version: 1.2.3\n"
                "Tested software version: 4.5.6\n"
                "Build version: 2024.04.01"
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertIsNone(ext.app_version)

    def test_bmw_idc23_software_platform_versions_ignored(self) -> None:
        issue = _issue(
            summary="[IDC23] crash on bench",
            description=(
                "BMW IDC23 software version: 03.62.20\n"
                "Target I-Step: 24.07.508\n"
                "I-Step: 24.07.500\n"
                "OS Version: 13\n"
                "Android version: 13\n"
                "Platform version: 03.62.20\n"
                "Vehicle software version: 24.07"
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertIsNone(ext.app_version)

    def test_version_requires_app_or_package_anchor(self) -> None:
        # A bare "Version: X.Y.Z" without the v<code> (...) envelope or
        # an app-version label must NOT be promoted.
        issue = _issue(
            summary="[IDC23] crash",
            description=(
                "Version: 3.0.0.1 was tested.\n"
                "v4.5.6 was also tested.\n"
                "Crashed at 9.9.9 randomly."
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertIsNone(ext.app_version)

    # ------------------------------------------------------------------
    # v2: anchored version forms.
    # ------------------------------------------------------------------

    def test_package_line_with_parens_extracts_version(self) -> None:
        # "Package: <pkg> v<code> (<X.Y.Z>)" is the canonical app-anchored
        # version line in BMW-style crash reports.
        issue = _issue(
            summary="crash",
            description=(
                "Process: com.partner.x\n"
                "Package: com.partner.x v100 (1.2.3)"
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "com.partner.x")
        self.assertEqual(ext.app_version, "1.2.3")

    def test_version_line_with_parens_extracts_version(self) -> None:
        # Standalone "Version: v<code> (<X.Y.Z>)" line.
        issue = _issue(
            summary="crash",
            description=(
                "Process: com.partner.x\n"
                "Version: v100 (1.2.3)"
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_version, "1.2.3")

    def test_crash_id_ver_token_extracted(self) -> None:
        issue = _issue(
            summary="crash",
            description=(
                "Process: com.partner.x\n"
                "Crash ID: ABC123|ver=v203000001 (3.0.0.1)|other=stuff"
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_version, "3.0.0.1")

    def test_natural_language_app_version_phrase(self) -> None:
        # "<App> latest available version is installed X.Y.Z" pairs
        # an app name with a version simultaneously — both come back.
        issue = _issue(
            summary="some crash",
            description="Spotify latest available version is installed 8.5.6.",
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Spotify")
        self.assertEqual(ext.app_version, "8.5.6")

    # ------------------------------------------------------------------
    # v2: Package → app-name mapping.
    # ------------------------------------------------------------------

    def test_package_to_app_name_mapping_resolves_known_app(self) -> None:
        # The seeded catalog and JSON map both know Deezer.
        # Evidence may legitimately come from either source — the
        # catalog matches "deezer" inside "deezer.android.app" via
        # whole-word match before the package→app fallback even runs.
        issue = _issue(
            summary="[IDC23] crash due to deezer.android.app"
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "deezer.android.app")
        self.assertEqual(ext.app_name, "Deezer")
        # Catalog labels now embed the matched app, e.g.
        # ``catalog[Deezer]:summary``. Either source is fine — both
        # mean "the bot knew this app".
        evidence = ext.evidence["app_name"]
        self.assertTrue(
            evidence.startswith("catalog[")
            or evidence.startswith("package_app_map:"),
            f"Unexpected evidence label: {evidence!r}",
        )

    def test_unknown_package_yields_derived_app_name(self) -> None:
        # User-stated rule: "if package_name is identified, app_name
        # should follow" — even when the package isn't in the catalog,
        # the bot derives a best-effort name from the coordinate.
        issue = _issue(
            summary="[IDC23] crash due to com.unknown.partner.x"
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "com.unknown.partner.x")
        self.assertIsNotNone(ext.app_name)
        self.assertNotEqual(ext.app_name, "com.unknown.partner.x")

    # ------------------------------------------------------------------
    # v2: BMW-3370 full case — every signal at once.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # v3: Structured Jira custom fields take priority over regex.
    # ------------------------------------------------------------------

    def test_bmw_3362_structured_fields_drive_extraction(self) -> None:
        # The Jira ticket has structured custom fields populated:
        #   App Name      = Bild TV
        #   Package Name  = com.mekmedia.bild.auto
        #   Version Name  = 3.0.0.5 and 3.1.13
        # The summary contains an [IDC23] env tag and no app metadata
        # in free text — the bot must read the structured fields.
        issue = _issue(
            summary="[IDC23] crash report",
            description="See attached log",
            structured=StructuredFields(
                app_name="Bild TV",
                package_name="com.mekmedia.bild.auto",
                version_names=["3.0.0.5", "3.1.13"],
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Bild TV")
        self.assertEqual(ext.package_name, "com.mekmedia.bild.auto")
        self.assertEqual(ext.app_versions, ["3.0.0.5", "3.1.13"])
        self.assertEqual(ext.app_version, "3.0.0.5, 3.1.13")
        self.assertEqual(ext.confidence, "high")
        self.assertEqual(ext.evidence["app_name"], "jira_structured_field")
        self.assertEqual(ext.evidence["package_name"],
                         "jira_structured_field")
        self.assertEqual(ext.evidence["app_version"],
                         "jira_structured_field")

    def test_structured_field_overrides_disagreeing_regex(self) -> None:
        # Description has a regex-extractable phrase pointing at a
        # *different* package — the structured field MUST win.
        issue = _issue(
            summary="crash due to com.regex.fake",
            description=(
                "Process: com.regex.fake\n"
                "versionName=9.9.9\n"
            ),
            comments=[Comment(
                author="QA",
                body=("Spotify latest available version is "
                      "installed 9.9.9."),
                created="2026-04-30T09:00:00.000+0000",
            )],
            structured=StructuredFields(
                app_name="Bild TV",
                package_name="com.mekmedia.bild.auto",
                version_names=["3.0.0.5", "3.1.13"],
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "com.mekmedia.bild.auto")
        self.assertEqual(ext.app_name, "Bild TV")
        self.assertEqual(ext.app_versions, ["3.0.0.5", "3.1.13"])

    def test_structured_partial_regex_fills_gaps(self) -> None:
        # Structured field gives only the package; regex fills the
        # version from a versionName= label. The structured package
        # wins, even though regex also produced a (different) one.
        issue = _issue(
            summary="something broke",
            description=(
                "Process: com.regex.fake\n"
                "versionName=4.7.2\n"
            ),
            structured=StructuredFields(
                package_name="com.mekmedia.bild.auto",
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "com.mekmedia.bild.auto")
        self.assertEqual(ext.app_versions, ["4.7.2"])
        self.assertEqual(ext.evidence["package_name"],
                         "jira_structured_field")

    def test_structured_versions_field_with_jira_version_objects(self) -> None:
        # Some Jira fields come back as a list of {"name": "..."} objects;
        # the orchestrator hands them to the extractor already-parsed
        # via StructuredFields, so this test mirrors that contract.
        issue = _issue(
            summary="crash",
            structured=StructuredFields(
                app_name="Deezer",
                package_name="deezer.android.app",
                version_names=["3.0.0.5", "3.1.13", "4.0.0"],
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_version, "3.0.0.5, 3.1.13, 4.0.0")
        self.assertEqual(ext.app_versions, ["3.0.0.5", "3.1.13", "4.0.0"])

    def test_unity_version_ignored_even_with_structured_pkg_only(self) -> None:
        # The Jira description has Unity / SW / I-Step values that must
        # not be promoted, and the structured field has no version
        # so the result has no app_version.
        issue = _issue(
            summary="[IDC23] crash on bench",
            description=(
                "Unity Version: 2022.3.10\n"
                "SW Version: 03.62.20\n"
                "I-Step: 24.07.508\n"
                "Target I-Step: 24.07.508\n"
                "BMW IDC23 software version: 03.62.20\n"
            ),
            structured=StructuredFields(
                app_name="Bild TV",
                package_name="com.mekmedia.bild.auto",
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "com.mekmedia.bild.auto")
        self.assertEqual(ext.app_name, "Bild TV")
        self.assertIsNone(ext.app_version)
        self.assertEqual(ext.app_versions, [])

    # ------------------------------------------------------------------
    # v3: Package validation — false positives must be rejected.
    # ------------------------------------------------------------------

    def test_two_segment_domain_is_not_a_package(self) -> None:
        # "example.com" sits in the description; without an app-anchored
        # signal, the extractor must reject it.
        issue = _issue(
            summary="check example.com please",
            description=(
                "We pinged example.com and got a 503.\n"
                "See report at foo.org for details."
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertIsNone(ext.package_name)

    def test_url_is_not_a_package(self) -> None:
        issue = _issue(
            summary="crash on https://faurecia-aptoide.atlassian.net/foo",
            description=(
                "Saw the issue at "
                "https://faurecia-aptoide.atlassian.net/browse/X-1\n"
                "Reach out at user@example.com."
            ),
        )
        ext = RegexExtractor().extract(issue)
        # The URL/email pieces must NOT register as packages.
        self.assertIsNone(ext.package_name)

    # ------------------------------------------------------------------
    # v4: KnownAppCatalog handles natural-language "<app> <version>"
    # mentions (BMW-3402 case) and stricter bare-package validation
    # rejects JSON/XML key noise like "term.show_file".
    # ------------------------------------------------------------------

    def test_bmw_3402_zoom_in_summary_natural_language(self) -> None:
        # Real ticket: summary "Error code 16 in zoom 1.0.8.9" plus
        # comments mentioning Zoom and Zoom for Cars. No structured
        # fields, no package coordinate in any free text — but the
        # catalog knows Zoom.
        issue = _issue(
            summary="Error code 16 in zoom 1.0.8.9",
            description=(
                "platform: G70; /DE; i7 60 xD; ...\n"
                "Head Unit: IDCEVO25; BMW IDCEVO; ...\n"
            ),
            comments=[
                Comment(
                    author="Petro Martynenko",
                    body=(
                        "Zoom app issue with error popup, to be assigned to "
                        "Forvia. Previously also was reported on "
                        "Zoom for Cars 1.0.8.7"
                    ),
                    created="2026-04-30T09:00:00.000+0000",
                ),
                Comment(
                    author="Smart_Error_Management_",
                    body=(
                        "Defect name: Error code 16 shown when placing a "
                        "Zoom call from favorites. Found in Function: "
                        "CarPlay (score 6)"
                    ),
                    created="2026-04-30T09:30:00.000+0000",
                ),
            ],
        )
        ext = RegexExtractor().extract(issue)
        # The summary natural-language signal wins: app=Zoom, version=1.0.8.9.
        self.assertEqual(ext.app_name, "Zoom")
        self.assertIn("1.0.8.9", ext.app_version or "")
        # The Zoom catalog entry has THREE package variants — Forvia
        # Zoom for Cars (1.0.x), Stellantis Zoom (5.x) and Zoom for
        # Mercedes (6.7.x / 7.0.x). Version 1.0.8.9 disambiguates to
        # com.forvia.zoomapp (this is the bug we're fixing: the bot
        # was previously reporting us.zoom.videomeetings, which has a
        # totally different version line, 5.6.x).
        self.assertEqual(ext.package_name, "com.forvia.zoomapp")

    def test_term_show_file_from_json_attachment_is_rejected(self) -> None:
        # Reproduce the BMW-3402 false positive: a key like
        # "term.show_file" inside a JSON attachment must NOT be
        # promoted to package_name. Two-segment + JSON attachment +
        # no labeled context = nothing should fire.
        issue = _issue(
            summary="Error code 16 in zoom 1.0.8.9",
            attachments=[Attachment(
                filename="BRiAN_177535022_TraceLink.json",
                mime_type="application/json",
                size=200,
                content_url="https://example.invalid/x",
                text=(
                    '{"event": "term.show_file", "data": {'
                    '"path": "/var/log/x.log", "session_id": 12345}}'
                ),
            )],
        )
        ext = RegexExtractor().extract(issue)
        self.assertNotEqual(ext.package_name, "term.show_file")
        # Two-segment "data.input" should also NOT register.
        for noisy in ("term.show_file", "data.input", "data.path"):
            self.assertNotEqual(ext.package_name, noisy)

    def test_zoom_for_cars_specific_match_in_comment(self) -> None:
        # Comment mentions "Zoom for Cars 1.0.8.7" — catalog longest-
        # alias rule must resolve to "Zoom for Cars" not bare "Zoom".
        issue = _issue(
            summary="something broke",
            comments=[Comment(
                author="QA",
                body="Reported on Zoom for Cars 1.0.8.7 last week",
                created="2026-04-30T10:00:00.000+0000",
            )],
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Zoom for Cars")
        self.assertIn("1.0.8.7", ext.app_version or "")

    def test_two_segment_package_in_log_attachment_still_rejected(self) -> None:
        # Even in a .log/.txt attachment, a bare 2-segment dotted
        # token mustn't become the package — only ≥3-segment bare
        # hits are accepted (or any labeled hit at any segment count).
        log_text = "Some line with foo.bar mentioned and not labelled."
        issue = _issue(
            summary="x",
            attachments=[Attachment(
                filename="trace.log",
                mime_type="text/plain",
                size=len(log_text),
                content_url="https://example.invalid/x",
                text=log_text,
            )],
        )
        ext = RegexExtractor().extract(issue)
        self.assertNotEqual(ext.package_name, "foo.bar")

    def test_catalog_match_does_not_invent_unrelated_version(self) -> None:
        # If the catalog matches an app but no nearby version exists,
        # we must NOT pull a version from somewhere else in the text.
        # The version comes from the same window or stays null.
        issue = _issue(
            summary="Spotify",
            description=(
                "Below is some unrelated content with version 9.9.9 "
                "buried in a stack trace from a different module."
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Spotify")
        # 9.9.9 is far from "Spotify" (different source / line) so it
        # should NOT be paired up by the catalog scan.
        self.assertNotEqual(ext.app_version, "9.9.9")

    def test_bmw_3444_package_without_explicit_name(self) -> None:
        # BMW-3444 reproduction. The bot had package=de.tagesschau and
        # version=1.0.6 but app_name was "not detected". User rule:
        # one package = one app, so a name MUST follow.
        issue = _issue(
            summary="JAVA_CRASH crash due to de.tagesschau JAVA:4dbe…",
            description="versionName=1.0.6",
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "de.tagesschau")
        self.assertEqual(ext.app_version, "1.0.6")
        # Catalog now has Tagesschau → de.tagesschau.
        self.assertEqual(ext.app_name, "Tagesschau")

    def test_bmw_3441_apps_template_in_description(self) -> None:
        # BMW-3441 reproduction. Summary mentions "Tagesschau" but the
        # description's structured template explicitly says
        # "Tagesschau Automotive v1.0.6" — the more specific name must
        # win, and its package (de.tagesschau.automotive) must follow.
        issue = _issue(
            summary="Tagesschau Rendering is too small on RSE",
            description=(
                "Apps (if specific version, please mention): "
                "Tagesschau Automotive v1.0.6\n"
                "Actions:\n"
                "- Open Tagesschau Automotive\n"
                "Expected: The available videos should render correctly."
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Tagesschau Automotive")
        self.assertEqual(ext.app_version, "1.0.6")
        self.assertEqual(ext.package_name, "de.tagesschau.automotive")

    def test_bmw_3441_apps_template_without_version(self) -> None:
        # BMW-3440-style case: "Apps: Nextory" in description with no
        # explicit version. Bot must still resolve to (Nextory,
        # com.gtl.nextory).
        issue = _issue(
            summary="Nextory: Not possible to play content after login",
            description=(
                "SW: Latest SW\n"
                "Apps (if specific version, please mention): Nextory\n"
                "Actions: Open Nextory App in IDCevo"
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Nextory")
        self.assertEqual(ext.package_name, "com.gtl.nextory")

    def test_apps_template_outranks_summary_catalog(self) -> None:
        # When summary catalog finds "Tagesschau" and description
        # Apps template finds "Tagesschau Automotive", the more-
        # specific Apps template wins (it carries +8 bonus).
        issue = _issue(
            summary="Tagesschau crash",
            description=(
                "Apps (if specific version, please mention): "
                "Tagesschau Automotive v1.0.6"
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Tagesschau Automotive")
        self.assertEqual(ext.package_name, "de.tagesschau.automotive")

    def test_bmw_3438_spiegel_via_catalog_not_derived_automotive(self) -> None:
        # Reproduction of BMW-3438. The bot used to derive
        # "Automotive" from ``de.spiegel.android.automotive.mmo``
        # because the longest middle segment of the package
        # coordinate is "automotive". With Der Spiegel now in the
        # catalog the bot reports the correct name; if the catalog
        # ever loses that entry, the smarter derive picks "Spiegel"
        # (brand-first) instead of "Automotive".
        crash_dump = (
            "packageName=de.spiegel.android.automotive.mmo\n"
            "versionName=1.0.1\n"
            "processName=de.spiegel.android.automotive.mmo\n"
        )
        issue = _issue(
            summary=(
                "JAVA_CRASH crash due to "
                "de.spiegel.android.automotive.mmo "
                "JAVA:8ec0db76f500bc29f8350705ae1fad10e2c3b3ed"
            ),
            description="See attached crash dump",
            attachments=[Attachment(
                filename="data_app_crash@1776681160374.txt",
                mime_type="text/plain",
                size=len(crash_dump),
                content_url="x",
                text=crash_dump,
            )],
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(
            ext.package_name, "de.spiegel.android.automotive.mmo",
        )
        # The package is now in the catalog → app_name comes back
        # from catalog_by_package, NOT from derive, NOT "Automotive".
        self.assertEqual(ext.app_name, "Der Spiegel")
        self.assertNotEqual(ext.app_name, "Automotive")

    def test_smarter_derive_skips_mid_segment_noise(self) -> None:
        # Brand-first rule: leftmost meaningful segment after
        # stripping prefix/suffix/mid-segment noise.
        self.assertEqual(
            _derive_name_from_package(
                "de.spiegel.android.automotive.mmo"
            ),
            "Spiegel",
        )
        # Brand segment shorter than 4 chars is skipped; rightmost
        # meaningful segment wins.
        self.assertEqual(
            _derive_name_from_package("com.gtl.nextory"),
            "Nextory",
        )
        # Repeated segment beats brand-first.
        self.assertEqual(
            _derive_name_from_package(
                "com.radioline.android.radioline.auto"
            ),
            "Radioline",
        )

    def test_derived_name_caps_confidence_at_medium(self) -> None:
        # Even when all three fields are filled, a DERIVED name means
        # we never claim "high" confidence — the user explicitly said
        # high should require Admin Portal / catalog verification.
        # We pick a package that ISN'T in the catalog so derive runs.
        crash_dump = (
            "packageName=com.fictional.brand.app\n"
            "versionName=2.0.0\n"
        )
        issue = _issue(
            summary="JAVA_CRASH crash due to com.fictional.brand.app",
            attachments=[Attachment(
                filename="dump.txt",
                mime_type="text/plain",
                size=len(crash_dump),
                content_url="x",
                text=crash_dump,
            )],
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "com.fictional.brand.app")
        self.assertIsNotNone(ext.app_name)  # derived
        self.assertIn(
            "derived_from_package:",
            ext.evidence.get("app_name", ""),
        )
        # Cap: derived means at most "medium", never "high".
        self.assertNotEqual(ext.confidence, "high")

    def test_bmw_3459_audible_word_is_not_audible_app(self) -> None:
        # Real BMW-3459. The description has the adjective "audible"
        # ("Ringtone on MD2 and on IDCevo audible") which used to
        # match the Audible catalog entry. Description also explicitly
        # names "Zoom for Cars 1.0.8.9RC13". The bot must NOT pick
        # Audible.
        description = (
            "Pre-Condition: Mobile device 2 and HU (IDCevo) are logged "
            "in with same Zoom Account\n"
            "1. Start Call from MD1 to MD2 → Ringtone on MD2 and on "
            "IDCevo audible\n"
            "Tested SW Version:\n"
            "Zoom app / 3rd Party App Version: Zoom for Cars 1.0.8.9RC13 "
            "(Car & Rack)"
        )
        issue = _issue(
            summary=(
                "[PEnt-Testevent I-518 VI][2607] After call switch from "
                "MD to HU missed call notification is shown on IDCevo "
                "and Calling icon/Widget is disappearing"
            ),
            description=description,
        )
        ext = RegexExtractor().extract(issue)
        # The Audible false-positive must not survive.
        self.assertNotEqual(ext.app_name, "Audible")
        self.assertNotEqual(ext.package_name, "com.audible.application")
        # The bot should resolve to Zoom for Cars + the BMW-specific
        # package variant + the actual version named in the text.
        self.assertEqual(ext.app_name, "Zoom for Cars")
        self.assertEqual(ext.app_version, "1.0.8.9RC13")
        self.assertEqual(
            ext.package_name, "com.forvia.zoomapp.push_notifications",
        )

    def test_requires_context_blocks_adjective_audible(self) -> None:
        # Direct test of the requires_context mechanism.
        from bot.catalog import KnownAppCatalog
        cat = KnownAppCatalog.load_default()
        # "audible" used as adjective — no context cue → no match.
        matches = cat.find_in_text(
            "the ringtone was audible from the next room"
        )
        names = [m.app.name for m in matches]
        self.assertNotIn("Audible", names)
        # "Audible app crashes" — "app" is a context cue → match accepted.
        matches = cat.find_in_text("the Audible app crashes on launch")
        names = [m.app.name for m in matches]
        self.assertIn("Audible", names)

    def test_bmw_3460_quoted_game_name_with_system_package_noise(
        self,
    ) -> None:
        # Real BMW-3460. The bot used to pick
        # ``com.bmwgroup.idnext.bmwcarplayinterface.service`` (a BMW
        # platform component, present in logs because CarPlay was
        # involved) and call the app "Bmwgroup". The actual subject
        # is the game "Unblock It" — package com.marketjs.unblockit.
        issue = _issue(
            summary=(
                'IDC_SI_Drive Test - phoneccall could not be '
                'established while game "unblock it" was played'
            ),
            description=(
                'start the game "unblock it" and get an incoming '
                'phone call via ACP'
            ),
            comments=[Comment(
                author="BMW_Interface_User",
                body=(
                    "callingPackageName: com.marketjs.unblockit "
                    "flags= 0 sdk= 30. Carplay disconnected: "
                    "com.bmwgroup.idnext.bmwcarplayinterface.service"
                ),
                created="2026-05-13T15:00:00.000+0000",
            )],
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Unblock It")
        self.assertEqual(ext.package_name, "com.marketjs.unblockit")
        # The BMW system service must NEVER be picked.
        self.assertNotEqual(
            ext.package_name,
            "com.bmwgroup.idnext.bmwcarplayinterface.service",
        )
        self.assertNotIn("Bmwgroup", ext.app_name)

    def test_bmw_system_packages_are_blocklisted(self) -> None:
        # Even when a labeled ``packageName=com.bmwgroup.*`` is the
        # only candidate, the bot must not surface it as the app
        # under test. With no other valid package, package_name
        # remains None — better empty than wrong.
        issue = _issue(
            summary="some crash",
            description=(
                "packageName=com.bmwgroup.idnext.bmwcarplayinterface.service\n"
                "packageName=com.bmw.theme.efficient1\n"
                "packageName=com.bmw.android.providers.settings.car"
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertIsNone(ext.package_name)

    def test_quoted_app_name_in_app_context_is_captured(self) -> None:
        # "app \"Spotify\" crashed" → catalog finds Spotify directly,
        # but the quoted extractor is the path that fires when an
        # app isn't in the catalog yet.
        issue = _issue(
            summary='the app "Unblock It" crashes on launch',
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Unblock It")

    def test_bmw_inventory_pairs_package_with_version(self) -> None:
        # BMW STAT_APPS_TEXT inventory format. The parser pairs the
        # package with the version on the same line; the summary
        # anchor then keeps only the entry belonging to the
        # summary's app family.
        inventory = (
            "STAT_APPS_TEXT_4 ...;"
            "com.bmwgroup.apinext.cdcsync.service;14;010109 00:00/"
            "com.forvia.zoomapp.push_notifications;1.0.8.9RC11;270426 06:09/"
            "com.radioline.android.radioline.auto;2.4.1;270426 08:29/"
            "de.tagesschau.automotive;1.0.6;270426 06:15/"
        )
        issue = _issue(
            summary=(
                "Zoom: even though Gallery is marked, "
                "need to activate again"
            ),
            attachments=[Attachment(
                filename="CW30641_20260427_112722_CheckIn.txt",
                mime_type="text/plain",
                size=len(inventory),
                content_url="x",
                text=inventory,
            )],
        )
        ext = RegexExtractor().extract(issue)
        # The Zoom inventory entry's package + version make it
        # through the anchor; Radioline/Tagesschau are dropped.
        self.assertEqual(ext.app_name, "Zoom")
        self.assertEqual(
            ext.package_name, "com.forvia.zoomapp.push_notifications",
        )
        self.assertEqual(ext.app_version, "1.0.8.9RC11")
        # Evidence trail shows the inventory pairing for both fields.
        self.assertIn(
            "bmw_inventory[com.forvia.zoomapp.push_notifications]",
            ext.evidence["package_name"],
        )
        self.assertIn(
            "bmw_inventory[com.forvia.zoomapp.push_notifications]",
            ext.evidence["app_version"],
        )

    def test_bmw_inventory_drops_version_when_paired_pkg_not_in_anchor(
        self,
    ) -> None:
        # If the inventory only mentions apps OUTSIDE the summary
        # family, no version is reported — better empty than wrong.
        inventory = (
            "com.radioline.android.radioline.auto;2.4.1;270426 08:29/"
            "de.tagesschau.automotive;1.0.6;270426 06:15/"
        )
        issue = _issue(
            summary=(
                "Zoom: passcode not visible on keypad "
                "for the IDC23 head unit"
            ),
            attachments=[Attachment(
                filename="CheckIn.txt", mime_type="text/plain",
                size=len(inventory), content_url="x", text=inventory,
            )],
        )
        ext = RegexExtractor().extract(issue)
        # No Zoom-family entry in this trimmed inventory, so
        # the bot reports no version (rather than borrowing
        # Radioline's 2.4.1 or Tagesschau's 1.0.6).
        self.assertEqual(ext.app_name, "Zoom")
        self.assertIsNone(ext.app_version)
        self.assertNotEqual(
            ext.package_name, "com.radioline.android.radioline.auto",
        )

    def test_bmw_3454_summary_anchor_rejects_inventory_noise(self) -> None:
        # Reproduction of BMW-3454. The summary is unambiguously
        # about Zoom. An attached ``CheckIn.txt`` is a system-wide
        # app inventory listing dozens of installed apps including
        # Radioline 2.4.1, Tagesschau Automotive 1.0.6, Spotify, …
        # Without the summary anchor, the bot picked
        # ``com.radioline.android.radioline.auto`` and reported the
        # ticket as a Radioline issue.
        checkin_excerpt = (
            "STAT_APPS_TEXT_8  ...;"
            "com.bmwgroup.apinext.entertainment.maestro;2.4.5;"
            "...;com.radioline.android.radioline.auto;2.4.1;270426 08:29;"
            "...;com.forvia.youtube;2.0.68;270426 06:07;"
            "...;com.smokoko.careatscar3_appning;1.1.102;270426 06:23;"
            "...;de.tagesschau.automotive;1.0.6;270426 06:15;"
            "...;com.forvia.zoomapp.push_notifications;1.0.8.9RC11;270426 06:09;"
        )
        issue = _issue(
            summary=(
                "Zoom: even though Gallery is marked, "
                "need to activate again"
            ),
            description=(
                "1. start zoom call\n"
                "2. gallery is chosen, thats why galery mode is expected\n"
                "3. 'Sprecher' is active, even though 'Galery' is marked..."
            ),
            attachments=[Attachment(
                filename="CW30641_20260427_112722_CheckIn.txt",
                mime_type="text/plain",
                size=len(checkin_excerpt),
                content_url="https://example.invalid/x",
                text=checkin_excerpt,
            )],
        )
        ext = RegexExtractor().extract(issue)
        # The summary anchor restricts package candidates to the Zoom
        # family. com.forvia.zoomapp.push_notifications is in
        # com.forvia.zoomapp's family (dotted-prefix); Radioline,
        # Tagesschau, Smokoko, YouTube are NOT.
        self.assertEqual(ext.app_name, "Zoom")
        self.assertTrue(
            (ext.package_name or "").startswith("com.forvia.zoomapp"),
            f"Expected a com.forvia.zoomapp.* package, got "
            f"{ext.package_name!r}",
        )
        # Radioline's 2.4.1 (which used to win) must not be reported.
        self.assertNotEqual(ext.app_name, "Radioline")
        self.assertNotEqual(ext.app_version, "2.4.1")
        self.assertNotEqual(
            ext.package_name, "com.radioline.android.radioline.auto",
        )

    def test_summary_anchor_does_not_fire_when_summary_has_no_known_app(
        self,
    ) -> None:
        # Sanity: if the summary doesn't catalog-match anything, the
        # anchor is a no-op and the original extraction logic runs.
        issue = _issue(
            summary="Generic crash on bootup",
            description=(
                "Process: com.gtl.nextory\n"
                "versionName=2.4.9\n"
            ),
        )
        ext = RegexExtractor().extract(issue)
        # Without a summary anchor, the labeled package + the catalog
        # entry for Nextory still produce a clean result.
        self.assertEqual(ext.package_name, "com.gtl.nextory")
        self.assertEqual(ext.app_name, "Nextory")

    def test_summary_anchor_keeps_family_variants(self) -> None:
        # The anchor must INCLUDE family-related entries — a summary
        # mentioning bare "Tagesschau" must still allow
        # ``Tagesschau Automotive`` (de.tagesschau.automotive) to win
        # when the Apps template inside the description names it
        # explicitly. This is BMW-3441's pattern again.
        issue = _issue(
            summary="Tagesschau Rendering is too small on RSE",
            description=(
                "Apps (if specific version, please mention): "
                "Tagesschau Automotive v1.0.6"
            ),
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.app_name, "Tagesschau Automotive")
        self.assertEqual(ext.package_name, "de.tagesschau.automotive")

    def test_bmw_3455_labeled_package_wins_over_catalog_derived(self) -> None:
        # Real BMW-3455 failure. Summary names "Zoom"; a Sherlog
        # comment contains many literal occurrences of
        # ``packageName=com.forvia.zoomapp.rsedemo`` (the specific
        # sub-variant). The bot used to pick the catalog-derived
        # ``com.forvia.zoomapp`` because the catalog hit scored
        # higher. With the lowered catalog-derived score, the
        # labeled hit wins, and the more-specific package is
        # reported.
        sherlog_excerpt = (
            "Zoom for Cars app debug info\n"
            "[16:28:03.000045] ECU=CDEA APP=ABKB CTX=LCAT | "
            "[latin-speller][2845]:(SpellerInputMethodService.kt:406) "
            ".onStartInput(): packageName=com.forvia.zoomapp.rsedemo "
            "actionId=0 label=null\n"
            "[16:29:09.825531] ECU=CDEA APP=ABKB CTX=LCAT | "
            "[latin-speller][2845]:(SpellerInputMethodService.kt:479) "
            ".onStartInputView(): com.forvia.zoomapp.rsedemo actionId=0\n"
        )
        issue = _issue(
            summary=(
                "Zoom: Entering passcode is not intuitive - the input "
                "is not shown on the keypad and cannot be changed"
            ),
            description=(
                "Download ZOOM via App Store.\n"
                "Start login process via mail address and password and "
                "additional passcode."
            ),
            comments=[Comment(
                author="BMW_Interface_User",
                body=sherlog_excerpt,
                created="2026-05-12T15:00:00.000+0000",
            )],
        )
        ext = RegexExtractor().extract(issue)
        # The labeled `packageName=` hit must win over the catalog-
        # derived `com.forvia.zoomapp` from the "Zoom for Cars" alias.
        self.assertEqual(
            ext.package_name, "com.forvia.zoomapp.rsedemo",
        )
        # App name is still Zoom (from summary + catalog) — the
        # catalog entry for Zoom owns this sub-variant package via
        # the consistency rule.
        self.assertEqual(ext.app_name, "Zoom")

    def test_longer_package_wins_on_tied_scores(self) -> None:
        # Synthetic version of the same property. The labeled hit
        # for a sub-variant must beat the more generic catalog
        # derivation even when scores would tie.
        issue = _issue(
            summary="Tagesschau crash",
            description=(
                "packageName=de.tagesschau\n"
                "packageName=de.tagesschau.automotive\n"
            ),
        )
        ext = RegexExtractor().extract(issue)
        # Two labeled hits, same score → longer (more specific) wins.
        self.assertEqual(ext.package_name, "de.tagesschau.automotive")

    def test_bmw_3401_radioline_consistency_filter(self) -> None:
        # Real failure case: the summary names the package but no
        # structured fields are populated. An attached log lists
        # multiple "Entertainment App Versions" — including
        # "YouTube 2.0.68" — far away from the actual app under test.
        # Without the consistency filter the bot was reporting
        # app_name=YouTube + version=2.0.68 next to package=Radioline.
        log_text = (
            "Entertainment App Versions:\n"
            "MediaApp 11.0.6\n"
            "YouTube 2.0.68\n"
            "Maestro 2.4.2\n"
            "Radioline 2.4.9\n"
        )
        issue = _issue(
            summary=(
                "[cde-mainline_26w17.7-1] TOMBSTONE crash due to "
                "com.radioline.android.radioline.auto"
            ),
            description=(
                "Issue created by Stability TraceDB analysis team.\n"
                "Process: com.radioline.android.radioline.auto\n"
                "Error Details: TOMBSTONE crash detected.\n"
            ),
            attachments=[Attachment(
                filename="trace.log",
                mime_type="text/plain",
                size=len(log_text),
                content_url="https://example.invalid/x",
                text=log_text,
            )],
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(
            ext.package_name, "com.radioline.android.radioline.auto",
        )
        # The consistency rule rejects "YouTube" because YouTube's
        # only catalog package is com.google.android.youtube — not
        # the Radioline package we just identified. The catalog also
        # has Radioline registered, so we end up reporting the right
        # app name.
        self.assertEqual(ext.app_name, "Radioline")
        self.assertNotEqual(ext.app_name, "YouTube")
        # The version paired with YouTube in the log (2.0.68) MUST be
        # dropped — its catalog source app disagrees with the chosen
        # final app.
        self.assertNotEqual(ext.app_version, "2.0.68")
        # Radioline's matching version (2.4.9 from the same log)
        # remains valid because its catalog source app matches.
        self.assertEqual(ext.app_version, "2.4.9")

    def test_consistency_filter_rejects_wrong_app_version(self) -> None:
        # Synthetic version of the consistency rule: text contains
        # "Spotify 8.5.6" but the package signal says Deezer. The
        # catalog ver_candidate from "Spotify 8.5.6" must NOT survive.
        issue = _issue(
            summary="crash due to deezer.android.app",
            description="Note: Spotify 8.5.6 is also installed.",
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "deezer.android.app")
        self.assertEqual(ext.app_name, "Deezer")
        self.assertNotEqual(ext.app_version, "8.5.6")

    def test_derive_name_always_produces_something_for_real_packages(self):
        # v5 policy (user-stated): when a package is identified, an
        # app name MUST follow. ``_derive_name_from_package`` is the
        # offline approximation of an Admin Portal lookup — it
        # should always pick a sensible segment for real Android
        # package coordinates.
        self.assertEqual(
            _derive_name_from_package(
                "com.radioline.android.radioline.auto"
            ),
            "Radioline",
        )
        self.assertEqual(
            _derive_name_from_package("de.tagesschau"), "Tagesschau",
        )
        # ``automotive`` is a platform-variant suffix — stripped, so
        # both Tagesschau variants derive the same display name (the
        # Admin Portal disambiguates the package).
        self.assertEqual(
            _derive_name_from_package("de.tagesschau.automotive"),
            "Tagesschau",
        )
        self.assertEqual(
            _derive_name_from_package("com.gtl.nextory"), "Nextory",
        )
        # Inputs that aren't proper packages still return None.
        self.assertIsNone(_derive_name_from_package(""))
        self.assertIsNone(_derive_name_from_package("notapackage"))
        self.assertIsNone(_derive_name_from_package(None))

    def test_bmw_3370_full_deezer_extraction(self) -> None:
        description = (
            "Process: deezer.android.app\n"
            "Package: deezer.android.app v203000001 (3.0.0.1)\n"
            "Version: v203000001 (3.0.0.1)\n"
            "Unity Version: 2022.3.10\n"
            "BMW IDC23 software version: 03.62.20\n"
            "Target I-Step: 24.07.508\n"
        )
        comment = Comment(
            author="QA",
            body="Deezer latest available version is installed 3.0.0.1.",
            created="2026-04-30T10:00:00.000+0000",
        )
        issue = _issue(
            summary=(
                "[IDC23] JAVA_CRASH crash due to "
                "[deezer.android.app](http://deezer.android.app) "
                "JAVA:bf4e0dac5925c4016c0dd02d2379ed8e0767029e"
            ),
            description=description,
            comments=[comment],
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "deezer.android.app")
        self.assertEqual(ext.app_version, "3.0.0.1")
        self.assertEqual(ext.app_name, "Deezer")
        self.assertEqual(ext.confidence, "high")

    def test_no_signal_returns_low_confidence(self) -> None:
        ext = RegexExtractor().extract(
            _issue(summary="App doesn't work", description="Please fix.")
        )
        self.assertIsNone(ext.package_name)
        self.assertIsNone(ext.app_version)
        self.assertIsNone(ext.app_name)
        self.assertEqual(ext.confidence, "low")

    def test_comment_provides_package_when_description_does_not(self) -> None:
        issue = _issue(
            summary="Crash report",
            description="Logs incoming.",
            comments=[Comment(
                author="QA Bot",
                body="Found it: applicationId=com.example.partnerapp",
                created="2026-04-30T09:00:00.000+0000",
            )],
        )
        ext = RegexExtractor().extract(issue)
        self.assertEqual(ext.package_name, "com.example.partnerapp")
        self.assertTrue(ext.evidence["package_name"].startswith("comment:"))

    def test_blocklist_filters_android_packages(self) -> None:
        desc = "Stack: at android.app.ActivityThread, at androidx.core.X"
        issue = _issue(summary="x", description=desc)
        ext = RegexExtractor().extract(issue)
        # Both candidates are filtered → no package detected.
        self.assertIsNone(ext.package_name)


class MergeTests(unittest.TestCase):

    def test_merge_keeps_regex_when_present_uses_claude_for_gaps(self) -> None:
        primary = Extraction(
            app_name=None, app_version="1.0.0",
            package_name="com.example.x", confidence="medium",
            evidence={"package_name": "attachment:log.txt"},
            backend="regex",
        )
        secondary = Extraction(
            app_name="ExampleX", app_version="9.9.9",
            package_name="com.fake.bad", confidence="high",
            evidence={"rationale": "from summary"},
            backend="claude",
        )
        merged = _merge(primary, secondary)
        self.assertEqual(merged.package_name, "com.example.x")  # kept
        self.assertEqual(merged.app_version, "1.0.0")           # kept
        self.assertEqual(merged.app_name, "ExampleX")           # filled
        self.assertEqual(merged.backend, "regex+claude")
        # Confidence is the higher of the two when at least one field hit.
        self.assertEqual(merged.confidence, "high")


class AdminPortalValidationTests(unittest.TestCase):
    """End-to-end coverage of the strict Admin Portal gate.

    The bot is wired up with the LocalAdminPortalSnapshotResolver in
    main.py so these tests stand in for the real production behaviour.
    The user's mandate is explicit: no high-confidence Slack banner
    without portal cross-check.
    """

    def _extractor_with_snapshot(self):
        # Use the snapshot resolver directly — same shape main.py wires
        # up so production behaviour and tests stay aligned.
        from bot.admin_portal_resolver import LocalAdminPortalSnapshotResolver
        return RegexExtractor(
            app_name_resolver=LocalAdminPortalSnapshotResolver(),
        )

    def test_bmw_3453_streamingmedia_renamed_to_radio_format(self) -> None:
        # Real BMW-3453. Bot used to report "Streamingmedia" (the
        # publisher) with high confidence because that was the
        # catalog's canonical name. The Admin Portal calls this app
        # "Radio Format". Two fixes need to land together: catalog
        # entry corrected, AND the portal-renaming step in the
        # extractor actually runs.
        issue = _issue(
            summary=(
                "[IDC23] JAVA_CRASH crash due to "
                "it.streamingmedia.radioformat"
            ),
            description=(
                "Application: it.streamingmedia.radioformat\n"
                "App Version: 1.2.0"
            ),
        )
        ext = self._extractor_with_snapshot().extract(issue)
        self.assertEqual(ext.app_name, "Radio Format")
        self.assertNotEqual(ext.app_name, "Streamingmedia")
        self.assertEqual(
            ext.package_name, "it.streamingmedia.radioformat",
        )
        self.assertEqual(ext.app_version, "1.2.0")
        # The portal verified the package AND the version is on the
        # published list — ✅ banner is earned.
        self.assertEqual(ext.admin_portal_status, "verified")
        self.assertEqual(ext.confidence, "high")
        # Portal versions list is surfaced so the Slack message can
        # show "published versions: 1.2.0, 1.0.1, 1.0.0".
        self.assertIn("1.2.0", ext.admin_portal_versions)

    def test_portal_no_match_downgrades_confidence(self) -> None:
        # Bot extracts a package that exists nowhere in the Admin
        # Portal snapshot. With the strict gate, confidence drops to
        # medium and the Slack note will ask for manual verification.
        issue = _issue(
            summary="crash in com.unknown.partner.x",
            description=(
                "Application: com.unknown.partner.x\n"
                "Version: 1.2.3"
            ),
        )
        ext = self._extractor_with_snapshot().extract(issue)
        self.assertEqual(ext.admin_portal_status, "no_match")
        self.assertNotEqual(ext.confidence, "high")

    def test_portal_overrides_catalog_when_they_disagree(self) -> None:
        # Sanity check on the renaming path: the LOCAL catalog entry
        # may have a different name from the snapshot. The snapshot
        # is authoritative. We exercise this by faking a snapshot at
        # runtime that disagrees with the catalog for Deezer.
        import json
        import tempfile
        from bot.admin_portal_resolver import LocalAdminPortalSnapshotResolver

        fake_snapshot = {
            "apps": [{
                "name": "Deezer (Auto Edition)",
                "package": "deezer.android.app",
                "versions": ["3.0.0.1"],
            }],
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False,
        ) as f:
            json.dump(fake_snapshot, f)
            path = f.name

        ext = RegexExtractor(
            app_name_resolver=LocalAdminPortalSnapshotResolver(path=path),
        ).extract(_issue(
            summary="Deezer crashes on launch",
            description=(
                "Application: deezer.android.app\n"
                "Version: 3.0.0.1"
            ),
        ))
        # Portal's authoritative name wins.
        self.assertEqual(ext.app_name, "Deezer (Auto Edition)")
        self.assertEqual(ext.admin_portal_status, "verified")

    def test_portal_version_mismatch_flags_status(self) -> None:
        # Snapshot has the package but only an older version. Bot
        # detects a build that's not on the published list. Portal
        # confirms package, denies version → version_mismatch.
        import json
        import tempfile
        from bot.admin_portal_resolver import LocalAdminPortalSnapshotResolver

        fake_snapshot = {
            "apps": [{
                "name": "Radio Format",
                "package": "it.streamingmedia.radioformat",
                "versions": ["1.0.0"],  # NOT 1.2.0
            }],
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False,
        ) as f:
            json.dump(fake_snapshot, f)
            path = f.name

        ext = RegexExtractor(
            app_name_resolver=LocalAdminPortalSnapshotResolver(path=path),
        ).extract(_issue(
            summary="Radio Format crashes",
            description=(
                "Application: it.streamingmedia.radioformat\n"
                "Version: 1.2.0"
            ),
        ))
        self.assertEqual(ext.admin_portal_status, "version_mismatch")
        # The bot still trusts the portal's app name (it's the same
        # package), but the version mismatch downgrades confidence.
        self.assertEqual(ext.app_name, "Radio Format")
        self.assertNotEqual(ext.confidence, "high")

    def test_bmw_3450_head_unit_build_id_does_not_leak_as_app_version(
        self,
    ) -> None:
        # Real BMW-3450. The ticket is about Zoom for Cars
        # (com.forvia.zoomapp.push_notifications). The summary
        # mentions a head-unit firmware build "2.2607.8-POINTFIX"
        # which the bot used to publish as the App Version with
        # confidence=medium. The Admin Portal has no such version
        # — its published list for this package is the 1.0.8.x
        # family. Mandate from the user: "if the three fields
        # aren't 100% aligned with the Admin Portal, do not
        # provide them."
        #
        # Expected new behaviour: the bogus version is SUPPRESSED.
        # The Slack message will show "App Version: not detected"
        # together with the portal's published list, which makes
        # it impossible for an assignee to take the wrong version
        # at face value.
        issue = _issue(
            summary=(
                "During an active Zoom call, a call is rejected and "
                "audio focus is lost"
            ),
            description=(
                "HU firmware: 2.2607.8-POINTFIX\n"
                "Package: com.forvia.zoomapp.push_notifications\n"
                "Steps: 1) Start Zoom call ..."
            ),
        )
        ext = self._extractor_with_snapshot().extract(issue)
        self.assertEqual(ext.app_name, "Zoom for Cars")
        self.assertEqual(
            ext.package_name, "com.forvia.zoomapp.push_notifications",
        )
        # The bogus head-unit build ID MUST NOT be reported as the
        # app version. Suppression is mandatory.
        self.assertIsNone(ext.app_version)
        self.assertNotIn("2.2607.8", (ext.app_version or ""))
        # Status remains explicit so the Slack note can surface the
        # portal's published list to guide manual verification.
        self.assertEqual(ext.admin_portal_status, "version_mismatch")
        self.assertIn("1.0.8.9RC13", ext.admin_portal_versions)
        # Confidence must not be high — the version isn't verified.
        self.assertNotEqual(ext.confidence, "high")

    def test_version_rescue_picks_portal_listed_candidate(self) -> None:
        # Ticket text has BOTH a head-unit build ID and a real
        # Zoom version. The bot's "best" candidate happens to be
        # the head-unit ID (scored higher by some heuristic), but
        # the rescue logic scans the other candidates and finds
        # the portal-listed Zoom version, swapping it in.
        issue = _issue(
            summary=(
                "Zoom for Cars 1.0.8.9RC13 — call rejected on HU "
                "build 2.2607.8-POINTFIX"
            ),
            description=(
                "Package: com.forvia.zoomapp.push_notifications\n"
                "App version: 1.0.8.9RC13\n"
                "HU firmware: 2.2607.8-POINTFIX"
            ),
        )
        ext = self._extractor_with_snapshot().extract(issue)
        self.assertEqual(ext.app_name, "Zoom for Cars")
        # Either the best candidate was already 1.0.8.9RC13 (good),
        # or the rescue swapped it in. Either way, the final
        # version must be one the portal knows.
        self.assertEqual(ext.app_version, "1.0.8.9RC13")
        self.assertEqual(ext.admin_portal_status, "verified")

    def test_portal_accepts_rc_suffix_against_base_version(self) -> None:
        # Real BMW-3459 shape. Snapshot has "1.0.8.9RC13"; ticket
        # also reports "1.0.8.9RC13". Exact match → verified.
        # (Earlier ticket versions might say "1.0.8.9" while portal
        # publishes "1.0.8.9RC13"; the prefix rule accepts both
        # directions.)
        import json
        import tempfile
        from bot.admin_portal_resolver import LocalAdminPortalSnapshotResolver

        fake_snapshot = {
            "apps": [{
                "name": "Zoom for Cars",
                "package": "com.forvia.zoomapp.push_notifications",
                "versions": ["1.0.8.9RC13"],
            }],
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False,
        ) as f:
            json.dump(fake_snapshot, f)
            path = f.name

        ext = RegexExtractor(
            app_name_resolver=LocalAdminPortalSnapshotResolver(path=path),
        ).extract(_issue(
            summary="Zoom for Cars 1.0.8.9RC13 crash",
            description=(
                "Application: com.forvia.zoomapp.push_notifications\n"
                "Version: 1.0.8.9RC13"
            ),
        ))
        self.assertEqual(ext.app_name, "Zoom for Cars")
        self.assertEqual(ext.app_version, "1.0.8.9RC13")
        self.assertEqual(ext.admin_portal_status, "verified")


if __name__ == "__main__":
    unittest.main()
