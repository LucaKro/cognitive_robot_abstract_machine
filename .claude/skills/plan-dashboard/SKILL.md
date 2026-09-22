---
name: plan-dashboard
description: Publish a live status dashboard Artifact for a multi-PR/multi-session initiative tracked under .claude/personal/plans/<plan-id>/plan.yaml on the personal-notes branch, cross-checked against live GitHub PR/CI/review state. Invoke as "/plan-dashboard <plan-id>" for one plan, or "/plan-dashboard" with no argument to publish the master index of every plan. Use when the user asks to see, refresh, or generate a plan dashboard, or asks "what's the status of <plan>".
allowed-tools: Bash, Read, Write, Artifact, AskUserQuestion, Skill
---

# Plan Dashboard

Validation, drift detection and rendering live in the `basstler` package; this
skill fetches the inputs, runs the scripts, and calls `Artifact`. What the page
shows, and why, is described in `.claude/hooks/README.md`. The schema is in
`plan-schema.md` next to this file.

Pick a scratch directory the `Artifact` tool can publish from (your session's
scratchpad, or one under the working directory) and call it `<out>` below.

## 1. Fetch the plan

```bash
source .claude/hooks/resolve-personal-notes-config.sh
git fetch "${NOTES_REMOTE}" "${NOTES_BRANCH}" --quiet
git show "FETCH_HEAD:${PLANS_DIR}/<plan-id>/plan.yaml" > /tmp/plan.yaml
git show "FETCH_HEAD:${PLANS_DIR}/<plan-id>/roadmap.md" > /tmp/roadmap.md
git show "FETCH_HEAD:${DASHBOARD_URL_CACHE_PATH}" > /tmp/dashboard-urls.yaml 2>/dev/null || true
```

If the plan does not exist, list `git ls-tree -r --name-only FETCH_HEAD "${PLANS_DIR}"`
and stop. If the notes branch itself is missing, offer `/setup-personal-notes`.

With no plan id (the master index), do this for every
`${PLANS_DIR}/*/plan.yaml`, each into its own scratch files.

## 2. Build it

```bash
python3 -m "${PULL_REQUEST_STATE_MODULE}" --plan /tmp/plan.yaml --output /tmp/pr_data.json
bash "${REFRESH_DASHBOARD_SCRIPT}" --plan-id <plan-id> \
  --plan /tmp/plan.yaml --roadmap /tmp/roadmap.md --pr-data /tmp/pr_data.json \
  --output <out>/dashboard.html \
  --tracking-url "https://github.com/<default_repository>/issues/<tracking_issue>"
```

Omit `--tracking-url` if the plan has no `tracking_issue`. GitHub redirects
`/issues/<n>` to the pull request when the number is one, so the same URL works
when the mailbox is a draft pull request.

The script auto-corrects items GitHub reports merged to `done` and pushes that,
then renders the page and writes its stylesheet and script beside it. It prints
one JSON summary - keep it for step 4; do not open the HTML. A non-zero exit
means the manifest failed validation: report its stderr to the user rather than
patching around it.

Master index: build each plan as above, then write one entry per plan (`id`,
`title`, `description`, `done`/`total` from its summary, `dashboard_url` from the
cache or `null`) into `/tmp/plans.json` and run
`python3 -m "${BUILD_INDEX_MODULE}" --plans /tmp/plans.json --output <out>/index.html`.

## 3. Publish

- **`url`:** the cache entry for `<plan-id>` (or `_index`). Never omit it when
  the cache has one - that mints a duplicate artifact which cannot be deleted
  from a session.
- **Supporting files:** the summary's `assets` are the stylesheet and script,
  named by their content. On a first publish, pass both in `files` (published
  name → `<out>/<name>`). On a republish, list the artifact's files
  (`Artifact`, `action: "list"`, `scope: "files"`, `url`) and pass only the
  assets not already there - usually none.
- **Favicon:** 📋 for a plan dashboard, 🗂️ for the index, on the first publish
  only.

Then record the URL with the script - never by writing the cache yourself.
List your artifacts (`action: "list"`), write the rows to
`/tmp/artifact_listing.json` as `[{"title", "url", "updated"}, ...]`, and run:

```bash
python3 -m "${RECORD_DASHBOARD_URL_MODULE}" --key <plan-id or _index> \
  --expected-title "<the plan's title, or the index page's>" \
  --listing /tmp/artifact_listing.json \
  --cache /tmp/dashboard-urls.yaml --output /tmp/updated-dashboard-urls.yaml
```

If it prints `"changed": true`, push the result:

```bash
bash "${WRITE_PERSONAL_NOTES_FILE_SCRIPT}" --source /tmp/updated-dashboard-urls.yaml \
  --destination "${DASHBOARD_URL_CACHE_PATH}" \
  --message "Record dashboard URL for <plan-id or _index>"
```

A non-zero exit means nothing was published under that title: report it.

## 4. Report

From the summary: counts by status, every item auto-corrected to done (by
name), every remaining drift flag, and every item ready to review - then the
link. If the master index exists and you refreshed a single plan, say it was
not refreshed; do not republish it unasked.
