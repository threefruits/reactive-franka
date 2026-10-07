# Reactive Franka

A small, uv-managed PyBullet project: Franka grasps a cup handle from above while the cup moves along X (±10 cm),
places it at the opposite station, returns home, and repeats. The cup reaches 5.5 cm/s,
matching the red obstacle (about nine times its previous circular speed). Two obstacles move back and
forth along independent straight-line paths: the original red obstacle on the table
and a blue box suspended at 43 cm, moving ±12 cm along Y at X = 62 cm. Blue is 12 cm
closer to the working area than before; yellow has been removed. Both appear in the
RGB-D map and contact reporting.
cuRobo **v2** consumes a live RGB-D TSDF/ESDF for collision
costs; fixed-camera YOLO segmentation and model-based ICP feed a Kalman pose tracker.

## Install

Linux, Python 3.11, an NVIDIA CUDA GPU, and a compatible driver are required for MPC.
The lockfile pins cuRobo's v2 revision and PyTorch 2.7.1/CUDA 12.6. The first GPU run
compiles CUDA/Warp kernels and takes longer than subsequent runs.

```bash
uv sync --extra gpu --extra vision
```

## Run

First verify the simulator, ICP, mapping, and MPC with an explicit simulator instance mask:

```bash
uv run --extra gpu reactive-franka --perception sim-mask
```

The trained YOLO weights are included at `weights/cup-handle-seg.pt` and selected by
default. Run from the repository root; no retraining is needed:

```bash
uv run --extra gpu --extra vision reactive-franka
```

Optional: regenerate the training data and retrain for a changed scene or camera:

```bash
uv run --extra vision python -m reactive_franka.train --output outputs/training-handle
uv run --extra gpu --extra vision reactive-franka \
  --weights outputs/training-handle/cup-seg/weights/best.pt
```

The training command generates 200 labeled synthetic images with tabletop and lifted cup positions,
varied colors, orientations, close-up handle grasps, moving obstacles, and negative examples, then trains for 30
epochs. Simulator masks are **training labels only** in this mode. Inference uses RGB
YOLO masks and depth; it never falls back to simulator cup labels. The stock
`yolo11n-seg.pt` model can be selected with `--weights`, but does not reliably detect
this stylized cup. The first use downloads pretrained weights from Ultralytics.

GUI is the default. Stop with Ctrl-C. For bounded headless verification:

```bash
uv run --extra gpu --extra vision reactive-franka --headless --fast \
  --steps 2400 --cycles 2
```

`--steps 0` runs continuously. The loop logs **pick → place → go_home → pick**.
`--cycles` counts complete pick/place/home cycles; the count increases only after the
hand returns to its startup position and orientation. Home is a Cartesian MPC target,
not a joint reset. A bounded run exits with an error if it cannot complete the cycles. `--fast` removes wall-clock pacing;
all moving-object trajectories still use simulation time. Normal pacing is best effort,
not a hard real-time guarantee. Outputs go to `outputs/` (override with `--output`):
`camera.png`, `handle-grasp.png` (latest successful grasp), `telemetry.jsonl`, and `summary.json`.

A lightweight CPU diagnostic is also available. It has **no obstacle avoidance or MPC**:

```bash
uv run reactive-franka --controller bullet-ik --perception sim-mask \
  --headless --fast --steps 1800 --cycles 2
```

## Data flow

```text
PyBullet RGB-D ─┬─ RGB → YOLO boxes + instance mask → depth cloud → upright ICP
               │                                                   ↓
               │                                   Kalman pose + velocity estimate
               │                                                   ↓
               │                                        reactive pick/place target
               └─ self/target filtering → cuRobo TSDF → ESDF          ↓
                                                        └──────→ cuRobo v2 MPC
                                                                 ↓
                                              measured joints ← ideal joint servo in Bullet
```

- Physics: 240 Hz; motor/control updates: 30 Hz; RGB-D, perception, and mapping: 10 Hz
  in simulation time. MPC reads Bullet joint position and velocity on every solve.
  The arm uses an ideal position servo that executes the four MPC samples over eight
  physics ticks, with acceleration carried from the previous MPC action. Joint velocity
  limits are scaled to 25%. Gripper motors and released-object dynamics use Bullet.
  This avoids conflating trajectory planning with tuning a torque/position controller.
- The external RGB-D camera is fixed at **(0.30, −0.75, 1.40) m**, looking at
  **(0.48, 0, 0.12) m**. This elevated oblique view was selected by comparing visibility
  through the grasp at both stations and the line-motion endpoints. It does not follow
  the arm or cup. Change `Camera`'s `eye`/`target` defaults in `camera.py` to reposition it;
  retrain YOLO when changing viewpoints. Intrinsics and optical-to-world extrinsics are calibrated from PyBullet's
  projection/view matrices. Depth is optical-axis distance in meters, not raw OpenGL
  depth-buffer values. PyBullet/SciPy use XYZW quaternions; cuRobo uses WXYZ.
- A 1 cm TSDF and 2 cm ESDF cover the work area. Weight decay removes old obstacle
  surfaces as fresh frames arrive. MPC uses a 3.5 cm collision activation margin.
  ESDF storage stays stable for CUDA graph replay.
  The adapter corrects fractional voxel-count metadata in the pinned cuRobo revision:
  a nominal 60-cell dimension otherwise becomes 59.999996 and is truncated by the
  collision kernel. A GPU test verifies both the stride and a known collision-free pose.
  The obstacle pose is never passed to MPC as a privileged geometric primitive.
- The robot is removed from depth using simulator body IDs (a simulation self-filter).
  The detected target mask and a small region around its estimated position are removed
  so the gripper can approach it. The region includes the handle and follows the same
  bounded prediction window as the grasp through brief occlusions. During transport,
  the attached cup is also self-filtered using its simulator ID. A real camera setup
  needs robot/payload model rendering or another calibrated self-filter instead.
- ICP uses a known metric cup model and an upright-cup prior: it estimates XYZ
  and yaw; roll and pitch are fixed. A single view does not reliably determine yaw
  when the handle is hidden. Visible depth points outside the body radius provide
  handle yaw evidence; frames without that evidence update XYZ only. Every valid
  frame still contributes to the Kalman filter.
- The top-down grasp pinches the outer handle arc at cup-local **(58, 0, 5) mm**.
  Estimated yaw rotates this offset and aligns the finger closing axis across the
  handle plane. Yaw is held during the final approach while translation keeps tracking;
  each pickup first acquires yaw with uncertainty below 0.3 rad and retains it through occlusion. Placement includes the same
  handle offset so the cup center lands on the station. Lift, transfer, and retreat
  use a 65 cm hand height to clear the nearby airborne obstacle.
- `tracking.py` implements a constant-velocity Kalman filter for world-frame
  `[x, y, z, yaw, vx, vy, vz, yaw_rate]`. It predicts at every 30 Hz control tick.
  **Every distinct valid camera/ICP observation updates the filter at 10 Hz in every
  task phase**, including grasp, transport, release, and return home. It is not reset
  at phase transitions. No simulator cup pose or known motion trajectory is fused.
  ICP quality controls measurement noise; yaw residuals wrap at ±π, and Joseph-form
  covariance updates preserve numerical stability. Roll and pitch remain constrained
  by the upright model. Detection association prevents jumping to a distant obstacle
  mislabeled as a cup; estimates outside the workspace, insufficient depth, and poor
  ICP fits are not measurements.
- The high approach follows the Kalman estimate. Before approaching the cup, missing
  observations (older than 0.8 s or position uncertainty above 4 cm) pause within
  `pick`. Descent and closing continue following the estimate with a 0.4 s planar velocity
  lead to catch the faster cup. Short gripper occlusions use Kalman predictions for
  up to 2 s since the last observation; longer gaps lift the hand to clear the camera
  view while staying in `pick`. The filter receives observations throughout. Transport
  and return home use their own waypoints. There is no separate
  `observe` phase; internal gripper/waypoint steps appear in telemetry as `task_step`.
  Mapping and MPC collision avoidance stay active during every step, including home.
- Telemetry includes raw observed position, filtered position/yaw, velocity, covariance
  standard deviations, observation age, and update count. The summary reports updates
  by task phase, so measurement contributions during transport are visible.

## Validation

Verified locally on an RTX 4090: the complete cuRobo + TSDF/ESDF + trained YOLO + ICP +
Kalman pipeline completed **four top-down handle pick/place/home cycles in 67.7 simulated
seconds with zero recorded obstacle contacts**. All 495 valid camera observations updated
Kalman tracking: 271 during pick, 170 during place, and 54 during return home.
Eighteen tests pass, including handle collision geometry, rejection of body/open/misaligned
grasps, rotated-handle pickup and placement, fixed-camera cup visibility at both stations
and travel endpoints, XYZ-only updates when handle yaw is hidden, moving-target tracking,
cycle ordering, and the CUDA mapping regression.
Run artifacts are under `outputs/handle-grasp/`, including `handle-grasp.png` captured at
successful attachment. The validated weights for this camera are bundled at
`weights/cup-handle-seg.pt`. Training runs, datasets, and simulation outputs remain
generated and gitignored; the bundled weights are tracked in Git.

## Scope and limitations

This is a simulation integration demo, not a force-closure grasp benchmark. The cup's
collision shape combines a solid cylinder body with a ring of 12 capsules for the
handle; rendering/ICP use a hollow cup with a handle. A fixed constraint represents
attachment only after the closed gripper is near the handle (12/10/12 mm local XYZ
limits), points down within 15°, and aligns its closing axis within 20°. The old
body-center grasp no longer qualifies. This remains a simplified grasp latch. Release is
physical and placement is checked before the kinematic line motion restarts. The
line starts at the released position with zero offset, avoiding a restart position jump.

The MPC collision model covers the robot; the carried cup is not added to its collision
spheres in this minimal version. The transfer waypoint provides clearance above the table and airborne obstacles.
A single depth camera cannot reconstruct hidden geometry, and decay can erase unseen
obstacles. Neither this nor local MPC guarantees a collision-free solution in arbitrary
scenes. `collision_steps` reports robot contact with either moving obstacle, not a general
safety certificate. The simulator-trained YOLO model is specific to this scene.

## Development

```bash
uv run --group dev pytest -q
uv run --group dev ruff check src tests
# GPU regression (voxel layout, camera/robot FK agreement, live map updates):
RUN_GPU_TESTS=1 uv run --extra gpu --group dev pytest -q
```

If your shell injects ROS packages through `PYTHONPATH`, run tests with
`env -u PYTHONPATH uv run --group dev pytest -q` to avoid unrelated ROS pytest plugins.

Files: `simulation.py` builds the scene; `camera.py` handles RGB-D calibration;
`perception.py` implements segmentation and ICP; `tracking.py` fuses camera observations
with Kalman predictions; `control.py` integrates cuRobo;
`task.py` contains task phases; `main.py` runs and records the loop; `train.py` builds
synthetic training data and fine-tunes YOLO.

References: [cuRobo live volumetric MPC](https://nvlabs.github.io/curobo/latest/reference/live_volumetric_mapping_mpc.html),
[cuRobo installation](https://nvlabs.github.io/curobo/latest/getting-started/installation.html),
[Ultralytics segmentation](https://docs.ultralytics.com/tasks/segment/).
