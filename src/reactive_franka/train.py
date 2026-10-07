"""Generate labeled simulator images and adapt YOLO to the simple rendered cup.

Simulator labels are used offline for training, never as YOLO inference results.
"""

import argparse
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

from .geometry import handle_grasp_pose
from .simulation import HAND_TO_GRASP, HOME, Simulation


def main():
    import cv2
    import yaml
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("outputs/training"))
    parser.add_argument("--weights", default="yolo11n-seg.pt")
    args = parser.parse_args()
    if args.images < 20 or args.epochs < 1:
        parser.error("Need at least 20 images and one epoch")
    rng = np.random.default_rng(42)
    root = (args.output / "dataset").resolve()
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    sim = Simulation(False, args.output)
    try:
        for i in range(args.images):
            split = "val" if i % 5 == 0 else "train"
            position = [
                rng.uniform(0.40, 0.64),
                rng.choice([-1, 1]) * rng.uniform(0.17, 0.28),
                rng.uniform(0.14, 0.46) if i % 4 == 0 else 0.051,
            ]
            # Some negative frames teach the model that robot/table/obstacle are not cups.
            if i % 13 == 0:
                position = [2, 2, 0.05]
            yaw = rng.uniform(-np.pi, np.pi)
            p.resetBasePositionAndOrientation(
                sim.cup,
                position,
                p.getQuaternionFromEuler([0, 0, yaw]),
                physicsClientId=sim.client,
            )
            p.changeVisualShape(
                sim.cup, -1, rgbaColor=[*rng.uniform(0.45, 1, 3), 1], physicsClientId=sim.client
            )
            sim.move_obstacles(rng.uniform(0, 60))
            if i % 3 == 1 and i % 13:
                handle, orientation = handle_grasp_pose(position, yaw)
                goal = handle + [0, 0, HAND_TO_GRASP + rng.uniform(0, 0.25)]
                joints = p.calculateInverseKinematics(
                    sim.robot, sim.hand, goal, orientation, physicsClientId=sim.client
                )[:7]
            elif i % 3 == 2:
                goal = [rng.uniform(0.35, 0.6), rng.uniform(-0.35, 0.35), rng.uniform(0.16, 0.6)]
                joints = p.calculateInverseKinematics(
                    sim.robot, sim.hand, goal, [1, 0, 0, 0], physicsClientId=sim.client
                )[:7]
            else:
                joints = HOME
            for index, value in zip(sim.arm, joints):
                p.resetJointState(sim.robot, index, value, physicsClientId=sim.client)
            frame = sim.camera.capture(0)
            Image.fromarray(frame.rgb).save(root / "images" / split / f"{i:04d}.jpg")
            contours, _ = cv2.findContours(
                (frame.body_ids == sim.cup).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            label = ""
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 30:
                    points = contour.reshape(-1, 2) / [frame.rgb.shape[1], frame.rgb.shape[0]]
                    label = "0 " + " ".join(f"{x:.6f}" for x in points.ravel()) + "\n"
            (root / "labels" / split / f"{i:04d}.txt").write_text(label)
    finally:
        sim.close()
    dataset = root / "data.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "cup"}}
        )
    )
    model = YOLO(args.weights)
    model.train(
        data=str(dataset),
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project=str(args.output.resolve()),
        name="cup-seg",
        exist_ok=True,
        seed=42,
        deterministic=True,
        degrees=0,
        flipud=0,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=5,
        plots=False,
    )
    print(f"Run with --weights {args.output / 'cup-seg/weights/best.pt'}")


if __name__ == "__main__":
    main()
