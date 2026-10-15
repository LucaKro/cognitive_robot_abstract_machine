# CRAM TIAGo in RoboSimStudio

The TIAGo carries a milk carton from the kitchen counter to the kitchen island. CRAM plans
the task and drives the robot. [RoboSimStudio](https://github.com/robosimstudio/robosimstudio)
simulates it, using Newton with MuJoCo's own solver.

Nothing attaches the milk to the gripper. CRAM's physically simulated execution leaves
the milk to the physics, so the finger pads' friction alone holds it on the way to the
island.

## How it fits together

| Module | What it does |
| --- | --- |
| `descriptions.py` | Fetches the TIAGo and apartment descriptions at pinned commits, so no ROS workspace is needed. |
| `world.py` | Builds the semantic digital twin: the apartment, the milk and cereal on the counter, and the TIAGo. |
| `scene.py` | Exports the world to MJCF with `semantic_digital_twin`'s `MujocoBuilder`, then applies this demo's physics choices (see below). |
| `simulation.py` | `RoboSimStudioSimulation`, which `coraplex.execution_environment.PhysicallySimulatedRobot` steps in lockstep with the controller. |
| `plan.py` | `MilkTransport`, the plan: park, raise the torso, pick the milk up, carry it over and put it down. |
| `recording.py` | A video rendered by RoboSimStudio, and RoboSimStudio's viewer in the browser. |
| `demo.py` | The command line. |

On every controller tick (10 ms):

1. Giskard writes its command into the world state.
2. The simulation hands every servo the position the world holds for its joint.
3. RoboSimStudio advances the physics.
4. The milk's and the cereal's simulated poses are written back into the world.

## Running it

You need Python 3.12 and a checkout of this repository with the workspace packages
installed (`uv sync` at the repository root). Then install the demo's extra
dependencies:

```bash
pip install -r experiments/src/experiments/robosimstudio_tiago_demo/requirements.txt
```

This installs RoboSimStudio at the commit the demo was made with. It is kept out of the
workspace's own dependencies because it is used only here. Then run:

```bash
python -m experiments.robosimstudio_tiago_demo.demo --video milk.mp4
```

The first run fetches the robot and apartment descriptions, about 400 MB, to
`~/.cache/robosimstudio_tiago_demo`. To fetch them ahead of time:

```bash
python -m experiments.robosimstudio_tiago_demo.descriptions
```

The run writes the exported scene, and the video if you asked for one, to
`robosimstudio_tiago_demo_output/`. At the end it prints where the milk came to rest
compared with where the plan put it down.

| Option | Meaning |
| --- | --- |
| `--video FILE` | Render the run into `FILE`, inside the output directory: an overview next to a close-up, in real time. |
| `--viewer-port PORT` | Serve RoboSimStudio's viewer on `http://localhost:PORT` while the plan runs. |
| `--output-directory DIR` | Where the scene and the video go. |
| `--cache-directory DIR` | Where the descriptions are fetched to. |

The run uses MuJoCo's CPU solver, so it needs no GPU. It is slower than real time: a run
of about a minute and a half of simulated time takes several times that on a laptop, and
recording a video adds to it.

## Physics choices of this demo

The world is exported as `semantic_digital_twin` describes it. On top of that export,
`scene.py` makes these choices for this demo; the numbers are fields of `PhysicsChoices`:

- **Servos.** Every joint the plan commands is driven by a MuJoCo `<position>` servo. The
  robot's links are gravity-compensated, because in reality the servos hold the arm's own
  weight.
- **Base.** The export leaves out the TIAGo's differential drive, so the scene moves the
  base on three servoed joints (x, y, yaw) instead. CRAM still plans the base as a
  differential drive.
- **Collisions.** The robot passes through everything except its finger pads, which use
  rubber-like friction. The apartment's parts don't collide with each other or with the
  pads. The groceries collide with both.
- **Groceries.** Each grocery collides as the box bounding its mesh, so the flat pads
  squeeze flat sides. It is drawn as its mesh, in its own color.
- **Frame-only links.** Robot links that only mark a frame get a negligible mass, instead
  of the 1 kg default for links whose description declares no inertial.
- **Single precision.** Newton rounds inertias to single precision, so inertias that don't
  clear the triangle inequality by a small margin are raised slightly.

`world.py` gives each grocery a realistic mass (a full one-litre milk carton, a 500 g
cereal box) and the inertia of a solid box filling its bounds. A body parsed from a bare
mesh otherwise gets the default inertial, 1 kg with a moment of 1 kg·m², and the finger
pads cannot turn that with the robot.

`plan.py` makes the TIAGo turn towards the island away from it. Its base turns on the
spot, and the milk it holds at its side sweeps a circle around it.

## Tests

```bash
pytest test/experiments_test/robosimstudio_tiago_demo_test
```

- The scene and description tests run wherever the robot and apartment descriptions
  resolve.
- The simulation tests need RoboSimStudio and are skipped without it.
- The end-to-end test is marked `slow`; run it with `-m slow`.
