## tracy-sensor-mapping (PR #27, draft, base main)

Plan item of articulated-manipulation-under-uncertainty. Resolve run, auto mode.
CI green on a730ab8c; stalled on the author's review (15 threads).

User decisions at resolve: split by layer; units with pint.
Plan (in progress):
1. semdt: generic `ForceTorqueSensor(Sensor)`; Tracy arms HasSensors[wrist F/T
   sensors at <side>_tool0]; Robotiq `ObjectDetectionStatus` next to the gripper.
   No pint field on semdt dataclasses (ORM maps every semdt dataclass).
2. experiments: signals keyed by the semdt parts (no TracySide/TracyJoint);
   ROS topics stay here (TracyDriverNamespace.of_part). Units via pint
   (application registry) + PintUnitJSONSerializer. Gripper current reported
   as register counts; arm effort in amperes (driver default).
3. Statistics: channel noise and sample interval as probabilistic_model
   GaussianDistribution (Dirac when constant); no #22 dependency.
4. Small threads: empty __init__, no module constants (ClassVar/enum), npt
   typing, generic SignalRecorder instead of Any; AGENTS.md rules for npt
   typing and module constants.
5. Replies on every thread; resolve only where acted on (not krrood/segmind/
   actuator questions). Update simulated-sensors notes, roadmap, PR body.
Still open after this: run the tool on the real Tracy.
Plan tooling: worktree of basstler_experiments in the scratchpad.
