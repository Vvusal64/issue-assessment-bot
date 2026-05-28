# Jira Issue Assessment Bot

Automation bot that monitors the **Issue Assessment** queue in Jira
(`type = "OEM App Bug" AND status = Open`), extracts missing application
metadata (app name, version, package name) from the issue's summary,
description, comments, and text attachments, and DMs the assignee on
Slack with the suggested values and a confidence score.

## Architecture

```
bot/
├── main.py          # CLI entry point + orchestrator
├── config.py        # Environment / config loading
├── jira_client.py   # Atlassian Jira Cloud REST client (retry-aware)
├── slack_client.py  # Slack Bot API client (DM by email lookup)
├── extractor.py     # Regex + heuristic extractor with Claude fallback
├── state.py         # Idempotency store (JSON file)
└── utils.py         # Logging, retry decorator, ADF→text helper
```

The pipeline is deliberately modular: each stage is a single class with
narrow responsibilities so it can be replaced (e.g. swap JSON state for
SQLite, or add a new extractor backend) without touching the others.

## Pipeline

1. `JiraClient.search_open_assessment_issues()` runs the JQL filter and
   returns assigned, unprocessed issues, expanded with their
   description, comments, and text-mode attachments.
2. `Extractor.extract(issue)` produces an `Extraction` containing
   `app_name`, `app_version`, `package_name`, `confidence`,
   `evidence`, and a list of unresolved fields.
   - First pass: regex heuristics across summary, description, comments,
     and attachments. Patterns are scored by source and explicitness
     (e.g. `versionName=` in a logcat is high; bare `1.2.3` in a
     comment is low).
   - Fallback pass (optional): if `ANTHROPIC_API_KEY` is set and
     confidence is `low` or any field is `null`, call Claude with a
     compact, structured prompt and ask it to return strict JSON.
     Refusal → keep the regex result.
3. `SlackClient.dm_assignee(...)` resolves the assignee's email to a
   Slack user ID via `users.lookupByEmail`, opens an IM channel via
   `conversations.open`, and posts a structured message.
4. `StateStore.mark_processed(...)` records the issue key and the
   extraction so re-runs skip it.

## Setup

```bash
cd jira-assessment-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in credentials
python run.py              # one-shot run
python run.py --dry-run    # extract only, no Slack delivery
python run.py --limit 5    # cap how many issues are processed
python run.py --reset KEY  # remove KEY from state (re-process it)
```

## Required environment variables

| Variable | Purpose |
| --- | --- |
| `JIRA_BASE_URL` | e.g. `https://faurecia-aptoide.atlassian.net` |
| `JIRA_EMAIL` | Atlassian account email used to authenticate |
| `JIRA_API_TOKEN` | API token from id.atlassian.com |
| `JIRA_JQL` | Optional override of the default JQL |
| `SLACK_BOT_TOKEN` | `xoxb-…` token with `chat:write`, `users:read`, `users:read.email`, `im:write` |
| `STATE_FILE` | Optional path; defaults to `./state.json` |
| `ANTHROPIC_API_KEY` | Optional; enables the Claude fallback extractor |
| `ANTHROPIC_MODEL` | Optional; defaults to `claude-sonnet-4-5` |
| `LOG_LEVEL` | Optional; defaults to `INFO` |

## Idempotency

`state.json` maps each processed Jira key to a record:
```json
{
  "BMW-3371": {
    "processed_at": "2026-04-30T10:42:11Z",
    "extraction": { "app_name": "...", "app_version": "...",
                    "package_name": "...", "confidence": "medium" },
    "slack_message_ts": "1714471331.001200"
  }
}
```
The bot only sends a notification once per issue. Use `--reset KEY` to
force a re-process during debugging.

## Failure handling

- Every Jira and Slack call is wrapped in `utils.retry()` with
  exponential backoff (1s → 2s → 4s, max 4 attempts) and only retries
  on transient HTTP errors (5xx, 429, network).
- A single issue's failure is logged and the bot moves on to the next
  one; the state file is only updated for successful deliveries.
- Slack lookups that fail (no Slack user matches the Jira email) are
  logged with a clear message and the issue is left unprocessed so the
  next run can retry.

## Future enhancements

The `extractor.py` module exposes an `ExtractorBackend` abstract base
class so additional strategies (full Claude reasoning, log-format-aware
parsers, learned models) can be plugged in without changing the
orchestrator. The state store records the extraction itself, which
makes it possible to later compare extractions against ground truth
once issues are resolved (i.e. learning from corrections).
