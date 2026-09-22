# Fetching live pull request data into pr_data.json

Referenced by `plan-dashboard/SKILL.md`, `plan-create/SKILL.md` and
`dependency-readiness.md`. One command, no GitHub calls of your own:

```bash
source .claude/hooks/resolve-personal-notes-config.sh
python3 -m "${PULL_REQUEST_STATE_MODULE}" --plan /tmp/plan.yaml --output /tmp/pr_data.json
```

It reads exactly the pull requests the plan's items name, keeps the four fields
the dashboard uses (`state`, `draft`, `merged_at` - always written, `null`
included - and label names), writes `/tmp/pr_data.json`, and prints one line:
`{"pull_requests": <written>, "not_found": ["owner/repo#n", ...]}`.

Do not open `/tmp/pr_data.json`; the summary line is all a session needs. A pull
request listed under `not_found` is left out of the file, which the dashboard
reports as `not_found` - mention it rather than working around it.

Credentials: `GH_TOKEN` or `GITHUB_TOKEN`, else a logged-in `gh`. If the command
fails for lack of one, say so and stop. Do not fall back to fetching pull
requests through MCP tools: their full representations, tens of kilobytes each,
are what made this step expensive.
