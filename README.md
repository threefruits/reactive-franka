# Reactive Franka

A small PyBullet demo managed with uv. Franka grasps a moving cup by its handle
from above, places it at the other station, returns home, and repeats.
The cup and two obstacles move back and forth.

- A fixed RGB-D camera observes the scene.
- YOLO segmentation and ICP estimate the cup pose. A Kalman filter tracks its motion.
- cuRobo v2 MPC uses a live TSDF/ESDF map for collision avoidance.

## Setup

Requires Linux, uv, Python 3.11, and an NVIDIA GPU with a CUDA-compatible driver.

```bash
uv sync --extra gpu --extra vision
```

## Run

Run from the repository root:

```bash
uv run --extra gpu --extra vision reactive-franka
```

The trained weights are included at `weights/cup-handle-seg.pt` and loaded by default.
The first run may take longer while CUDA kernels compile.

The GUI opens by default. The loop is **pick → place → go home → repeat**.
Press **Ctrl-C** to stop. Images and logs are saved to `outputs/`.

To run two cycles without a GUI:

```bash
uv run --extra gpu --extra vision reactive-franka --headless --fast \
  --steps 2400 --cycles 2
```

`--fast` runs without real-time pacing. `--steps` limits the number of control steps.
A cycle finishes after the robot returns home.

## Retrain (optional)

Retrain the model if you change the camera position or scene:

```bash
uv run --extra vision python -m reactive_franka.train --output outputs/training-handle
uv run --extra gpu --extra vision reactive-franka \
  --weights outputs/training-handle/cup-seg/weights/best.pt
```

This generates 200 labeled simulator images and trains YOLO for 30 epochs.

Reference: [cuRobo live mapping and MPC](https://nvlabs.github.io/curobo/latest/reference/live_volumetric_mapping_mpc.html).
