"""Tests for the Slack message renderer.

Covers the three-state status line (✅ / ⚠️ / ❌), the "Open in Jira"
link, and the "Related ongoing tickets" section (populated, empty,
omitted).
"""

from __future__ import annotations

import unittest

from bot.jira_client import RelatedIssue
from bot.slack_client import compute_status_line, render_message


COMPLETE_EMOJI = "✅"   # ✅
PARTIAL_EMOJI = "⚠️"  # ⚠️
NONE_EMOJI = "❌"       # ❌


class StatusLineTests(unittest.TestCase):

    def test_all_three_present_is_high_confidence(self) -> None:
        # ✅ requires all three fields AND confidence='high' AND the
        # Admin Portal cross-check must have returned 'verified'.
        # Without portal verification, a name that passed the
        # extractor's heuristics is still just an unconfirmed guess
        # (BMW-3453 case: bot said "Streamingmedia" with high
        # confidence — but the portal calls it "Radio Format").
        emoji, text = compute_status_line(
            app_name="Deezer", app_version="3.0.0.1",
            package_name="deezer.android.app",
            confidence="high",
            admin_portal_status="verified",
        )
        self.assertEqual(emoji, COMPLETE_EMOJI)
        self.assertIn("high confidence", text)
        self.assertIn("Admin Portal", text)

    def test_high_confidence_without_portal_check_is_partial(self) -> None:
        # Legacy callers / dry-runs that don't pass an Admin Portal
        # status must NOT get the green banner. The bot must have
        # independently verified the package against the portal
        # before we tell the assignee "trust me, this is right".
        emoji, text = compute_status_line(
            app_name="Deezer", app_version="3.0.0.1",
            package_name="deezer.android.app",
            confidence="high",
            admin_portal_status="not_checked",
        )
        self.assertEqual(emoji, PARTIAL_EMOJI)
        self.assertIn("partially", text)

    def test_high_confidence_with_portal_no_match_is_partial(self) -> None:
        # The BMW-3453 failure mode: bot extracts a package, slaps a
        # name on it via the catalog, but the package isn't actually
        # in the Admin Portal. Bot must downgrade.
        emoji, text = compute_status_line(
            app_name="Streamingmedia", app_version="1.2.0",
            package_name="it.streamingmedia.unknown",
            confidence="high",
            admin_portal_status="no_match",
        )
        self.assertEqual(emoji, PARTIAL_EMOJI)
        self.assertIn("partially", text)

    def test_all_three_but_low_confidence_is_partial(self) -> None:
        # BMW-3438-style: package + derived name + version, but
        # confidence is only "medium" because the name is a guess.
        emoji, text = compute_status_line(
            app_name="Spiegel", app_version="1.0.1",
            package_name="de.spiegel.android.automotive.mmo",
            confidence="medium",
            admin_portal_status="verified",
        )
        self.assertEqual(emoji, PARTIAL_EMOJI)
        self.assertIn("partially", text)

    def test_package_only_is_partial(self) -> None:
        emoji, text = compute_status_line(
            app_name="Blanco", app_version=None,
            package_name="eimycarolaym.blanco",
            confidence="medium",
            admin_portal_status="verified",
        )
        self.assertEqual(emoji, PARTIAL_EMOJI)
        self.assertIn("partially", text)

    def test_nothing_detected_is_failure(self) -> None:
        emoji, text = compute_status_line(
            app_name=None, app_version=None, package_name=None,
        )
        self.assertEqual(emoji, NONE_EMOJI)
        self.assertIn("not detected", text)


class RenderMessageTests(unittest.TestCase):

    def _bmw_3370_complete(self):
        return render_message(
            issue_key="BMW-3370",
            issue_url=(
                "https://faurecia-aptoide.atlassian.net/browse/BMW-3370"
            ),
            assignee_name="vusal.orujlu",
            summary=(
                "[IDC23] JAVA_CRASH crash due to "
                "[deezer.android.app](http://deezer.android.app)"
            ),
            app_name="Deezer",
            app_version="3.0.0.1",
            package_name="deezer.android.app",
            confidence="high",
            related_issues=[],
            admin_portal_status="verified",
            admin_portal_versions=["3.0.0.1", "2.9.0"],
        )

    def _vf_533_partial(self):
        return render_message(
            issue_key="VF-533",
            issue_url=(
                "https://faurecia-aptoide.atlassian.net/browse/VF-533"
            ),
            assignee_name="vusal.orujlu",
            summary=(
                "[VF9][VN] Can not change sound and source when change "
                "by MFS"
            ),
            app_name="Blanco",
            app_version=None,
            package_name="eimycarolaym.blanco",
            confidence="medium",
            related_issues=[
                RelatedIssue(
                    key="VF-520",
                    summary="Similar issue summary here",
                    url="https://faurecia-aptoide.atlassian.net/browse/VF-520",
                ),
                RelatedIssue(
                    key="VF-481",
                    summary="Similar issue summary here",
                    url="https://faurecia-aptoide.atlassian.net/browse/VF-481",
                ),
            ],
        )

    def test_complete_message_has_check_emoji_and_open_in_jira(self) -> None:
        msg = self._bmw_3370_complete()
        text = msg["text"]
        self.assertIn(COMPLETE_EMOJI, text)
        self.assertIn("Metadata detected with high confidence", text)
        # Both the issue line and the explicit Jira action link.
        self.assertIn("Issue: BMW-3370", text)
        self.assertIn(
            "Open in Jira: "
            "https://faurecia-aptoide.atlassian.net/browse/BMW-3370",
            text,
        )
        self.assertIn("App Name: Deezer", text)
        self.assertIn("App Version: 3.0.0.1", text)
        self.assertIn("Package Name: deezer.android.app", text)
        self.assertIn("Related ongoing tickets:", text)
        self.assertIn("None found", text)

    def test_partial_message_lists_related_tickets(self) -> None:
        msg = self._vf_533_partial()
        text = msg["text"]
        self.assertIn(PARTIAL_EMOJI, text)
        self.assertIn("manual validation required", text)
        self.assertIn(
            "Open in Jira: "
            "https://faurecia-aptoide.atlassian.net/browse/VF-533",
            text,
        )
        # Missing version is rendered as "not detected", not as a guess.
        self.assertIn("App Version: not detected", text)
        # Both related tickets show up with their summaries.
        self.assertIn("VF-520 — Similar issue summary here", text)
        self.assertIn("VF-481 — Similar issue summary here", text)

    def test_related_section_omitted_when_lookup_was_skipped(self) -> None:
        msg = render_message(
            issue_key="X-1", issue_url="https://example/X-1",
            assignee_name="someone",
            summary="x", app_name=None, app_version=None,
            package_name=None, confidence="low",
            related_issues=None,  # explicit "we couldn't search"
        )
        # When related_issues is None we DON'T render the section at all
        # (vs. an empty list which renders "None found").
        self.assertNotIn("Related ongoing tickets", msg["text"])

    def test_multi_version_display_string(self) -> None:
        # The orchestrator hands us a comma-joined display string for
        # multi-version cases (e.g. BMW-3362). The renderer prints it
        # verbatim — no further processing.
        msg = render_message(
            issue_key="BMW-3362",
            issue_url=(
                "https://faurecia-aptoide.atlassian.net/browse/BMW-3362"
            ),
            assignee_name="vusal.orujlu",
            summary="[IDC23] crash report",
            app_name="Bild TV",
            app_version="3.0.0.5, 3.1.13",
            package_name="com.mekmedia.bild.auto",
            confidence="high",
            related_issues=[],
            admin_portal_status="verified",
            admin_portal_versions=["3.0.0.5", "3.1.13"],
        )
        self.assertIn("App Version: 3.0.0.5, 3.1.13", msg["text"])
        self.assertIn(COMPLETE_EMOJI, msg["text"])

    def test_admin_portal_no_match_downgrades_banner(self) -> None:
        # BMW-3453 regression: bot has all three fields with high
        # confidence, but the package wasn't in the Admin Portal.
        # Banner must be ⚠️ and the body must explain the portal miss.
        msg = render_message(
            issue_key="BMW-3453",
            issue_url=(
                "https://faurecia-aptoide.atlassian.net/browse/BMW-3453"
            ),
            assignee_name="vusal.orujlu",
            summary="[IDC23] crash report",
            app_name="Streamingmedia",
            app_version="1.2.0",
            package_name="it.streamingmedia.unknown",
            confidence="high",
            related_issues=[],
            admin_portal_status="no_match",
        )
        self.assertIn(PARTIAL_EMOJI, msg["text"])
        self.assertIn("Package NOT found in Admin Portal", msg["text"])

    def test_admin_portal_verified_shows_versions(self) -> None:
        # When the portal verifies, the published-version list is
        # surfaced so the assignee can sanity-check against the
        # ticket's reported build.
        msg = render_message(
            issue_key="BMW-3453",
            issue_url=(
                "https://faurecia-aptoide.atlassian.net/browse/BMW-3453"
            ),
            assignee_name="vusal.orujlu",
            summary="[IDC23] crash report",
            app_name="Radio Format",
            app_version="1.2.0",
            package_name="it.streamingmedia.radioformat",
            confidence="high",
            related_issues=[],
            admin_portal_status="verified",
            admin_portal_versions=["1.2.0", "1.0.1", "1.0.0"],
        )
        self.assertIn(COMPLETE_EMOJI, msg["text"])
        self.assertIn("Verified", msg["text"])
        self.assertIn("1.2.0", msg["text"])

    def test_blocks_contain_status_header_and_open_in_jira(self) -> None:
        # Round-trip the rich-blocks structure we send to Slack and
        # check the high-level shape is what we expect.
        msg = self._bmw_3370_complete()
        block_texts = [
            (b.get("text") or {}).get("text", "")
            for b in msg["blocks"]
            if b.get("type") == "section"
        ]
        joined = "\n".join(block_texts)
        self.assertIn("Metadata detected with high confidence", joined)
        self.assertIn("Open in Jira", joined)


if __name__ == "__main__":
    unittest.main()
