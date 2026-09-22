## basstler_experiments: gh stack in cloud sessions

Done:
- install-gh-stack.sh verified in a real web session (fresh tools dir: gh
  download, gh-stack clone/Go build, local extension install). CLAUDE_ENV_FILE
  PATH line works: later Bash calls see the gh dir first.
- Fixed false "installed": gh 2.90 exits 0 for `gh stack` without the
  extension; probe is now `gh extension list` (ccdad8e9).
- stacks.md command names/flags all exist; local init/add/rebase --upstack/
  --continue/--abort/push work end to end.

Open:
- The session proxy blocks GraphQL. `gh stack link`, `submit`, and
  `checkout <PR>` of a stack not known locally need it, so they fail in cloud
  sessions. REST works, including `POST repos/{o}/{r}/stacks`
  (`pull_requests: [...]`). stacks.md/plan_stack.link_command need a
  cloud-session path (open PRs via REST/MCP, register the stack via REST).
  Waiting on the user's decision.
- `gh repo sync <fork>` not run (writes to the fork).
