"""YOLO instance mask -> metric depth cloud -> model-based, upright-cup ICP."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .camera import Frame
from .geometry import model_points


@dataclass
class Estimate:
    world_from_object: np.ndarray
    mask: np.ndarray
    timestamp: float
    rmse: float
    inlier_fraction: float
    yaw_observed: bool = True

    @property
    def position(self):
        return self.world_from_object[:3, 3]


def upright_icp(model, observed, initial, iterations=25, threshold=0.018):
    """Observed-to-model correspondences avoid fitting hidden surfaces to visible ones.

    The tabletop cup has an upright prior: optimize yaw and XYZ, not roll/pitch.
    Model coordinates and all distances are in meters.
    """
    transform = initial.copy()
    for _ in range(iterations):
        transformed = model @ transform[:3, :3].T + transform[:3, 3]
        distances, indices = cKDTree(transformed).query(observed)
        good = distances < threshold
        if good.sum() < 25:
            break
        source, target = transformed[indices[good]], observed[good]
        a, b = source.mean(0), target.mean(0)
        cross = (source[:, :2] - a[:2]).T @ (target[:, :2] - b[:2])
        u, _, vt = np.linalg.svd(cross)
        correction = np.eye(2)
        correction[-1, -1] = np.linalg.det(vt.T @ u.T)
        r = np.eye(3)
        r[:2, :2] = vt.T @ correction @ u.T
        delta = np.eye(4)
        delta[:3, :3], delta[:3, 3] = r, b - r @ a
        transform = delta @ transform
        if np.linalg.norm(delta - np.eye(4)) < 1e-5:
            break
    fitted = model @ transform[:3, :3].T + transform[:3, 3]
    distances, _ = cKDTree(fitted).query(observed)
    good = distances < threshold
    rmse = float(np.sqrt(np.mean(distances[good] ** 2))) if good.any() else float("inf")
    return transform, rmse, float(good.mean())


class Perception:
    def __init__(self, mode: str, cup_id: int, weights: str, confidence: float = 0.25):
        self.mode, self.cup_id, self.confidence = mode, cup_id, confidence
        self.model = model_points()
        self.last = None
        self.detector = None
        if mode == "yolo":
            from ultralytics import YOLO

            self.detector = YOLO(weights)
            if self.detector.task != "segment":
                raise ValueError("Use YOLO segmentation weights, e.g. yolo11n-seg.pt")

    def detect_mask(self, frame: Frame):
        if self.mode == "sim-mask":
            return frame.body_ids == self.cup_id
        # Ultralytics ndarray inputs are BGR. retina_masks returns original resolution.
        result = self.detector.predict(
            frame.rgb[..., ::-1].copy(),
            verbose=False,
            conf=self.confidence,
            retina_masks=True,
            imgsz=640,
        )[0]
        if result.masks is None or result.boxes is None:
            return None
        candidates = [
            i for i, cls in enumerate(result.boxes.cls.tolist()) if result.names[int(cls)] == "cup"
        ]
        if not candidates:
            return None
        # Associate detections with the previously observed cup, not a newly visible
        # obstacle that YOLO also labels "cup". The bound permits carried-cup motion.
        if self.last is not None and frame.timestamp - self.last.timestamp < 0.8:
            plausible = []
            distance_limit = 0.07 + 0.4 * (frame.timestamp - self.last.timestamp)
            for index in candidates:
                mask = result.masks.data[index].cpu().numpy() > 0.5
                points = frame.points(mask)
                if (
                    len(points) >= 40
                    and np.linalg.norm(np.median(points, axis=0) - self.last.position)
                    < distance_limit
                ):
                    plausible.append(index)
            candidates = plausible
            if not candidates:
                return None
        # Prefer confidence among associated candidates; never substitute simulator labels.
        index = max(candidates, key=lambda i: float(result.boxes.conf[i]))
        return result.masks.data[index].cpu().numpy() > 0.5

    def estimate(self, frame: Frame):
        mask = self.detect_mask(frame)
        if mask is None:
            return None
        points = frame.points(binary_erosion(mask, iterations=1))
        if len(points) < 40:
            return None
        points = points[:: max(1, len(points) // 900)]
        center = np.median(points, axis=0)
        # The fixed camera also observes the cup while it is lifted/transported.
        points = points[(points[:, 2] > 0.005) & (points[:, 2] < 0.85)]
        if len(points) < 40:
            return None
        initial = np.eye(4)
        initial[:3, 3] = [center[0], center[1], max(0.051, np.percentile(points[:, 2], 95) - 0.05)]
        if self.last is not None and frame.timestamp - self.last.timestamp < 0.5:
            # Keep a current-cloud seed as well: a carried cup can move farther than
            # ICP's correspondence radius between camera frames.
            seed = self.last.world_from_object.copy()
            seed[:3, 3] = initial[:3, 3]
            seeds = [self.last.world_from_object, seed]
        else:
            seeds = []
            for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                seed = initial.copy()
                seed[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
                seeds.append(seed)
        fits = [upright_icp(self.model, points, seed) for seed in seeds]
        transform, rmse, fraction = min(fits, key=lambda fit: fit[1] + 0.02 * (1 - fit[2]))
        if (
            rmse > 0.008
            or fraction < 0.65
            or not 0.025 < transform[2, 3] < 0.8
            or not 0.30 < transform[0, 3] < 0.78
            or abs(transform[1, 3]) > 0.42
        ):
            return None
        # The cylindrical body cannot constrain yaw. Use visible points beyond its
        # radius as handle evidence; absent handle evidence still contributes XYZ.
        radial = points[:, :2] - transform[:2, 3]
        distance = np.linalg.norm(radial, axis=1)
        handle = (
            (distance > 0.040)
            & (distance < 0.072)
            & (np.abs(points[:, 2] - transform[2, 3] - 0.005) < 0.032)
        )
        yaw_observed = False
        if handle.sum() >= 6:
            directions = radial[handle] / distance[handle, None]
            direction = directions.mean(axis=0)
            if np.linalg.norm(direction) > 0.95:
                yaw = np.arctan2(direction[1], direction[0])
                transform[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
                yaw_observed = True
        estimate = Estimate(transform, mask, frame.timestamp, rmse, fraction, yaw_observed)
        self.last = estimate
        return estimate
