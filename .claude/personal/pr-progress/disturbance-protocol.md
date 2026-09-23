## disturbance-protocol (PR #30, stack 29: #24 -> #28 -> #30)

Plan (roadmap `## \`disturbance-protocol\``): protocol + metrics in
experiments/articulated_manipulation. Prior error is belief-only (physics built
from the true scene, controller's world from a believed scene; MultiSim needs a
separate physics world). Disturbances act on physics only, triggered by sim time
or physical opening. Metrics: success, false success, time, recovery transitions
authored (declared by the task for now), cycle time via ControlLoopProfiler.
Error-level magnitudes: configurable placeholders (user decision).

Done: branch, draft PR, stack registration (REST /stacks/29/add), manifest.
Next: TDD - protocol + metrics tests (no MuJoCo), then episode runner; MuJoCo
tests are CI-only. Acceptance: stuck-drawer probe counted as false success.
