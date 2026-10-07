"""Run with `uv run --extra gpu --extra vision reactive-franka`."""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .control import BulletIK, CuroboMPC
from .perception import Perception
from .simulation import Simulation
from .task import GRASP_PREDICTION_SECONDS, PickPlace
from .tracking import PoseKalmanFilter

LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reactive Franka + Kalman-tracked cup + two moving obstacles"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--controller", choices=["curobo", "bullet-ik"], default="curobo")
    parser.add_argument("--perception", choices=["yolo", "sim-mask"], default="yolo")
    parser.add_argument("--weights", default="yolo11n-seg.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--steps", type=int, default=0, help="30 Hz control steps; 0 runs continuously"
    )
    parser.add_argument("--cycles", type=int, default=0, help="Stop after N pick/place/home cycles")
    parser.add_argument("--fast", action="store_true", help="Run without real-time pacing")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.steps < 0 or args.cycles < 0 or not 0 < args.confidence <= 1:
        parser.error("steps/cycles must be nonnegative and confidence in (0, 1]")
    return args


def run(args):
    sim = Simulation(not args.headless, args.output)
    task = PickPlace()
    tracker = PoseKalmanFilter()
    updates_by_phase = {}
    steps, detections, collisions = 0, 0, 0
    started = time.monotonic()
    controller = None
    last_mask = None
    interrupted = False
    error = None
    try:
        perception = Perception(args.perception, sim.cup, args.weights, args.confidence)
        frame = sim.camera.capture(sim.time)
        estimate = perception.estimate(frame)
        last_observation = estimate
        last_mask = estimate.mask if estimate is not None else None
        if args.controller == "curobo":
            controller = CuroboMPC(sim, frame, last_mask)
        else:
            LOG.warning("Bullet IK diagnostic mode: no MPC or obstacle avoidance")
            controller = BulletIK(sim)
        with (args.output / "telemetry.jsonl").open("w") as log:
            while not args.steps or steps < args.steps:
                tick_started = time.monotonic()
                # Camera + YOLO + ICP + mapping at 10 Hz; arm control and physics at 30/240 Hz.
                if steps % 3 == 0:
                    if steps > 0:
                        frame = sim.camera.capture(sim.time)
                        # Fuse the fixed camera's measurements in EVERY task phase.
                        estimate = perception.estimate(frame)
                    if estimate is not None:
                        detections += 1
                        last_observation = estimate
                        last_mask = estimate.mask
                    else:
                        last_mask = None
                    previous_updates = tracker.updates
                    tracked = tracker.advance(sim.time, estimate)
                    updates_by_phase[task.phase] = (
                        updates_by_phase.get(task.phase, 0) + tracker.updates - previous_updates
                    )
                    # Keep a small exclusion around the last *estimated* cup pose through
                    # brief gripper occlusion; do not fuse target fragments as obstacles.
                    if tracked is not None and (
                        tracked.is_fresh(sim.time)
                        or (
                            task.phase == "pick"
                            and task.step != "preapproach"
                            and 0
                            <= sim.time - tracked.observation_timestamp
                            <= GRASP_PREDICTION_SECONDS
                        )
                    ):
                        valid = frame.depth > 0
                        points = frame.points(valid)
                        near_target = np.all(
                            np.abs(points - tracked.position) < [0.08, 0.08, 0.08], axis=1
                        )
                        region = np.zeros_like(valid)
                        region[valid] = near_target
                        last_mask = region if last_mask is None else last_mask | region
                    # During transport self-filter the attached payload using simulator labels.
                    if sim.constraint is not None:
                        last_mask = frame.body_ids == sim.cup
                    controller.update_map(frame, last_mask)
                else:
                    tracked = tracker.advance(sim.time)
                previous_phase = task.phase
                goal, quat, grip = task.update(sim, tracked)
                if previous_phase == "pick" and task.phase == "place":
                    Image.fromarray(sim.camera.capture(sim.time).rgb).save(
                        args.output / "handle-grasp.png"
                    )
                command = controller.command(goal, quat)
                if not np.isfinite(command).all():
                    raise RuntimeError("Non-finite controller command")
                sim.step(command, grip)
                collisions += int(sim.collision_count() > 0)
                steps += 1
                if steps % 30 == 0:
                    record = {
                        "time": round(sim.time, 3),
                        "phase": task.phase,
                        "task_step": task.step,
                        "cycles": task.cycles,
                        "goal": goal.tolist(),
                        "goal_quaternion_xyzw": quat.tolist(),
                        "hand": sim.hand_pose()[0].tolist(),
                        "detections": detections,
                        "collision_steps": collisions,
                        "mpc_failures": getattr(controller, "failures", 0),
                        "kalman_updates": tracker.updates,
                        "observation_position": (
                            last_observation.position.tolist()
                            if last_observation is not None
                            else None
                        ),
                        "target_position": tracked.position.tolist()
                        if tracked is not None
                        else None,
                        "target_velocity": tracked.velocity.tolist()
                        if tracked is not None
                        else None,
                        "target_yaw": tracked.yaw if tracked is not None else None,
                        "target_position_std": (
                            np.sqrt(np.diag(tracked.covariance)[:3]).tolist()
                            if tracked is not None
                            else None
                        ),
                        "observation_age": (
                            sim.time - tracked.observation_timestamp
                            if tracked is not None
                            else None
                        ),
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    Image.fromarray(frame.rgb).save(args.output / "camera.png")
                if args.cycles and task.cycles >= args.cycles:
                    break
                if task.waiting_for_target and sim.time > 5 and steps % 150 == 0:
                    LOG.warning(
                        "Waiting for a reliable cup mask/ICP pose; inspect outputs/camera.png"
                    )
                if not args.fast:
                    time.sleep(max(0, 1 / 30 - (time.monotonic() - tick_started)))
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        Image.fromarray(sim.camera.capture(sim.time).rgb).save(args.output / "camera.png")
        summary = {
            "controller": args.controller,
            "grasp_mode": "handle_top_down",
            "camera_eye": sim.camera.pose[:3, 3].tolist(),
            "perception": args.perception,
            "steps": steps,
            "sim_seconds": round(sim.time, 3),
            "cycles": task.cycles,
            "detections": detections,
            "kalman_updates": tracker.updates,
            "updates_by_phase": updates_by_phase,
            "collision_steps": collisions,
            "phase": task.phase,
            "wall_seconds": round(time.monotonic() - started, 2),
            "interrupted": interrupted,
            "error": error,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        LOG.info("Run summary: %s", summary)
        sim.close()
    if args.cycles and task.cycles < args.cycles and not interrupted:
        raise RuntimeError(f"Only completed {task.cycles}/{args.cycles} requested cycles")
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("curobo").setLevel(logging.WARNING)
    run(parse_args())


if __name__ == "__main__":
    main()
