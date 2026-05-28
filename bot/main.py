"""CLI orchestrator.

Run modes:
  python -m bot.main                # process new issues, send Slack DMs
  python -m bot.main --dry-run      # extract only; print plan, no Slack
  python -m bot.main --limit 5      # cap the number of issues processed
  python -m bot.main --reset KEY    # remove KEY from the state file
  python -m bot.main --list-state   # show processed keys
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .admin_portal_resolver import (
    AdminPortalResolver,
    LocalAdminPortalSnapshotResolver,
)
from .app_name_resolver import (
    ChainedResolver,
    JsonFileAppNameResolver,
    default_resolver,
)
from .oem_portal_resolver import load_oem_resolvers_from_env
from .catalog import KnownAppCatalog
from .config import Config, ConfigError
from .extractor import (
    ClaudeExtractor,
    Extraction,
    HybridExtractor,
    RegexExtractor,
)
from .jira_client import JiraClient, JiraIssue, RelatedIssue
from .jira_fields import StructuredFieldReader
from .slack_client import SlackClient, SlackError, render_message
from .state import StateStore
from .utils import configure_logging


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="jira-assessment-bot",
        description=(
            "Scan the Jira Issue Assessment queue, extract app metadata, "
            "and DM the assignee on Slack."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline but do not send Slack messages or update the state file.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N issues this run.",
    )
    p.add_argument(
        "--reset",
        metavar="ISSUE_KEY",
        default=None,
        help="Remove ISSUE_KEY from the state file and exit.",
    )
    p.add_argument(
        "--list-state",
        action="store_true",
        help="List processed issue keys and exit.",
    )
    p.add_argument(
        "--no-claude",
        action="store_true",
        help="Disable the Claude fallback even if ANTHROPIC_API_KEY is set.",
    )
    return p.parse_args(argv)


def run(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)

    try:
        cfg = Config.from_env()
    except ConfigError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        return 2

    log = configure_logging(cfg.log_level)

    state = StateStore(cfg.state_file)

    if args.list_state:
        keys = state.all_keys()
        log.info("State file contains %d processed issue(s).", len(keys))
        for k in sorted(keys):
            entry = state.get(k) or {}
            ext = entry.get("extraction", {})
            print(
                f"{k}\t{entry.get('processed_at', '?')}"
                f"\t{ext.get('confidence', '?')}"
                f"\t{ext.get('package_name') or '-'}"
            )
        return 0

    if args.reset:
        ok = state.reset(args.reset)
        log.info("Reset %s: %s", args.reset, "ok" if ok else "not found")
        return 0 if ok else 1

    jira = JiraClient(
        base_url=cfg.jira_base_url,
        email=cfg.jira_email,
        api_token=cfg.jira_api_token,
    )
    field_reader = StructuredFieldReader(jira)
    slack = SlackClient(bot_token=cfg.slack_bot_token)

    # ------------------------------------------------------------------
    # Resolver chain
    #
    # Order is significant: the chain returns the first non-empty hit,
    # so the most-trusted backend goes first. Admin Portal (when
    # configured) is the back-office source of truth and can also
    # supply versions; the local JSON map is the fallback so the bot
    # still resolves common apps when the portal isn't reachable.
    # ------------------------------------------------------------------
    resolver_backends = []
    if cfg.admin_portal_url and cfg.admin_portal_token:
        resolver_backends.append(AdminPortalResolver(
            base_url=cfg.admin_portal_url,
            api_token=cfg.admin_portal_token,
        ))
        log.info("Admin Portal resolver (live): %s", cfg.admin_portal_url)
    else:
        log.info("Admin Portal resolver (live): not configured (set "
                 "ADMIN_PORTAL_URL / ADMIN_PORTAL_TOKEN to enable).")
    # Always include the local snapshot — it acts as the authoritative
    # offline stand-in for the Admin Portal so the strict validation
    # gate in the extractor has something to check against even when
    # the live API isn't reachable. When BOTH are present, the live
    # API wins (it's listed first); the snapshot fills in the gaps.
    snapshot_resolver = LocalAdminPortalSnapshotResolver()
    resolver_backends.append(snapshot_resolver)
    log.info(
        "Admin Portal resolver (snapshot): %d entries loaded.",
        len(snapshot_resolver),
    )
    # OEM-specific portals (BMW, Mercedes, …) are chained AFTER the
    # central Admin Portal but BEFORE the local JSON map. Each one
    # is opt-in via <OEM>_PORTAL_URL / <OEM>_PORTAL_TOKEN env vars.
    oem_resolvers = load_oem_resolvers_from_env()
    resolver_backends.extend(oem_resolvers)
    if not oem_resolvers:
        log.info(
            "OEM portal resolvers: none configured (set "
            "BMW_PORTAL_URL/_TOKEN, MERCEDES_PORTAL_URL/_TOKEN, … to "
            "enable per-OEM cross-checking)."
        )
    resolver_backends.append(JsonFileAppNameResolver())
    resolver = (
        ChainedResolver(*resolver_backends)
        if len(resolver_backends) > 1 else resolver_backends[0]
    )

    # Pre-load the catalog once so we don't re-read JSON for every issue.
    catalog = KnownAppCatalog.load_default()
    log.info("Catalog: %d known apps loaded.", len(catalog.apps))

    def _make_regex_extractor() -> RegexExtractor:
        return RegexExtractor(
            app_name_resolver=resolver, catalog=catalog,
        )

    extractor: object
    if cfg.anthropic_api_key and not args.no_claude:
        try:
            claude = ClaudeExtractor(
                api_key=cfg.anthropic_api_key,
                model=cfg.anthropic_model,
            )
            hybrid = HybridExtractor(claude=claude)
            # Replace the inner regex extractor with our resolver- and
            # catalog-aware one so the structured / catalog signal is
            # available even on the Claude path.
            hybrid._regex = _make_regex_extractor()
            extractor = hybrid
            log.info(
                "Extractor: regex+catalog+Claude (model=%s)",
                cfg.anthropic_model,
            )
        except Exception as exc:
            log.warning("Claude unavailable (%s); using regex only.", exc)
            extractor = _make_regex_extractor()
    else:
        extractor = _make_regex_extractor()
        log.info("Extractor: regex+catalog (no Claude)")

    log.info("Querying Jira: %s", cfg.jira_jql)

    try:
        issues = jira.search_assigned_open_issues(
            cfg.jira_jql,
            max_results=args.limit,
        )
    except Exception as exc:
        log.error("Jira search failed: %s", exc)
        return 3

    log.info("%d assigned issue(s) returned.", len(issues))

    sent = 0
    skipped = 0
    failed = 0

    for issue in issues:
        if state.is_processed(issue.key):
            log.info("Skipping %s (already processed).", issue.key)
            skipped += 1
            continue

        try:
            issue.structured = field_reader.read(
                (issue.raw or {}).get("fields") or {}
            )
            log.warning("[STRUCTURED READ] %s | %s", issue.key, issue.structured)
        except Exception as exc:
            log.warning(
                "Could not parse structured fields for %s: %s — continuing with regex only.",
                issue.key,
                exc,
            )

        try:
            extraction = (
                extractor.extract(issue)
                if hasattr(extractor, "extract")
                else issue
            )  # type: ignore
        except Exception as exc:
            log.exception("Extraction failed for %s: %s", issue.key, exc)
            failed += 1
            continue

        log.info(
            "Issue %s | confidence=%s | app=%s ver=%s pkg=%s",
            issue.key,
            extraction.confidence,
            extraction.app_name,
            extraction.app_version,
            extraction.package_name,
        )

        if args.dry_run:
            print(_format_dry_run(issue, extraction))
            continue

        ok = _deliver(slack, jira, state, issue, extraction)
        if ok:
            sent += 1
        else:
            failed += 1

    log.info("Done. sent=%d skipped=%d failed=%d", sent, skipped, failed)

    return 0 if failed == 0 else 4


def _deliver(
    slack: SlackClient,
    jira: JiraClient,
    state: StateStore,
    issue: JiraIssue,
    extraction: Extraction,
) -> bool:
    log = logging.getLogger("bot.deliver")

    if not issue.assignee_email:
        log.warning("Issue %s has no assignee email — cannot DM.", issue.key)
        return False

    related: Optional[List[RelatedIssue]] = []

    try:
        # Strictly same-package only — see build_related_open_jql for
        # the rationale. If we don't have a package, we don't surface
        # any related tickets (better empty than wrong).
        related = jira.find_related_open_issues(
            exclude_key=issue.key,
            package_name=extraction.package_name,
            max_results=5,
        )
    except Exception as exc:
        log.warning(
            "Related-ticket lookup for %s failed: %s — proceeding without that section.",
            issue.key,
            exc,
        )
        related = None

    note: Optional[str] = None
    missing = extraction.missing_fields()

    if missing:
        note = (
            "Manual validation may still be required — could not detect: "
            f"{', '.join(missing)}."
        )
    elif extraction.admin_portal_status == "no_match":
        note = (
            "Bot could not find this package in the Admin Portal — "
            "please verify the (app name / package / version) in the "
            "back-office before reporting to the partner."
        )
    elif extraction.admin_portal_status == "version_mismatch":
        note = (
            "Bot found the package in the Admin Portal but the version "
            "is not on the published list — confirm the build before "
            "reporting."
        )
    elif extraction.admin_portal_status == "not_checked":
        note = (
            "Bot could not consult the Admin Portal for this package — "
            "manual verification recommended."
        )
    elif extraction.confidence == "low":
        note = "Confidence is low — please confirm before reporting."

    msg = render_message(
        issue_key=issue.key,
        issue_url=issue.url,
        assignee_name=issue.assignee_display_name or "there",
        summary=issue.summary,
        app_name=extraction.app_name,
        app_version=extraction.app_version,
        package_name=extraction.package_name,
        confidence=extraction.confidence,
        version_code=extraction.version_code,
        environment_name=extraction.environment_name,
        related_issues=related,
        note=note,
        admin_portal_status=extraction.admin_portal_status,
        admin_portal_versions=extraction.admin_portal_versions,
    )

    try:
        user_id = slack.lookup_user_by_email(issue.assignee_email)
    except SlackError as exc:
        log.warning(
            "Slack user not found for %s (%s): %s — leaving %s unprocessed so a future run can retry.",
            issue.assignee_display_name,
            issue.assignee_email,
            exc,
            issue.key,
        )
        return False

    try:
        channel = slack.open_dm(user_id)
        result = slack.post_message(
            channel=channel,
            text=msg["text"],
            blocks=msg["blocks"],
        )
    except SlackError as exc:
        log.error("Slack delivery failed for %s: %s", issue.key, exc)
        return False

    state.mark_processed(
        issue.key,
        extraction=extraction.to_dict(),
        slack_message_ts=result.ts,
        slack_channel=result.channel,
    )

    log.info(
        "Notified %s about %s (channel=%s ts=%s)",
        issue.assignee_email,
        issue.key,
        result.channel,
        result.ts,
    )
    return True


def _format_dry_run(issue: JiraIssue, extraction: Extraction) -> str:
    return (
        f"\n--- DRY RUN: {issue.key} ---\n"
        f"Assignee: {issue.assignee_display_name} <{issue.assignee_email}>\n"
        f"Summary : {issue.summary}\n"
        f"app_name      = {extraction.app_name}\n"
        f"app_version   = {extraction.app_version}\n"
        f"package_name  = {extraction.package_name}\n"
        f"confidence    = {extraction.confidence}\n"
        f"backend       = {extraction.backend}\n"
        f"evidence      = {extraction.evidence}\n"
    )


if __name__ == "__main__":
    sys.exit(run())