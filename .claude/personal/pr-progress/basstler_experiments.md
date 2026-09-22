## basstler_experiments: gh stack in cloud sessions

Done:
- install-gh-stack.sh fixed (false "installed"; ccdad8e9) and verified live.
- Cloud sessions can't use GraphQL, so stacks go over REST (da427599):
  basstler.stack_registration registers an item's PR as the top layer
  (create/extend, idempotent). plan_stack lost its `link` command.
  stacks.md and kickoff use `gh api` for PRs, `gh stack init` to adopt a stack,
  REST merge-upstream for the fork. Verified live with a 3-layer stack
  (#18-#20, stack #21, since closed/unstacked).

Open:
- User must delete the leftover stack-test/{bottom,middle,top} branches on the
  fork (the proxy refuses branch deletion).
- Not live-tested: REST merge-upstream (blocked as a write to the fork's
  main), and GitHub retargeting the layers above one that lands.
