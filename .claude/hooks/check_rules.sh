#!/usr/bin/env bash
# Claude Code Stop hook: before Claude hands its work back, report the code it changed
# since the last commit that breaks a rule `krrood check` enforces, so it fixes it first.
set -u

hook_input=$(cat)
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# Without krrood in this environment there is nothing to check with; pre-commit and CI still do.
command -v krrood > /dev/null 2>&1 || exit 0

# Claude already went back once for this hook's report; stop rather than loop on it.
if grep -Eq '"stop_hook_active"[[:space:]]*:[[:space:]]*true' <<< "$hook_input"; then
    exit 0
fi

if report=$(krrood check --since HEAD 2>&1); then
    exit 0
fi
echo "The code changed since the last commit breaks rules of AGENTS.md:" >&2
echo "$report" >&2
exit 2
