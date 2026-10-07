"""PyBullet OpenGL depth -> calibrated optical (+X right,+Y down,+Z forward) RGB-D."""

from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    body_ids: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    timestamp: float

    def points(self, mask: np.ndarray) -> np.ndarray:
        v, u = np.nonzero(mask & np.isfinite(self.depth) & (self.depth > 0))
        z = self.depth[v, u]
        k = self.intrinsics
        xyz = np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z))
        t = self.world_from_camera
        return xyz @ t[:3, :3].T + t[:3, 3]


class Camera:
    """Fixed external RGB-D camera; view, intrinsics, and extrinsics never follow the robot."""

    def __init__(
        self,
        client: int,
        width: int = 640,
        height: int = 480,
        eye=(0.30, -0.75, 1.40),
        target=(0.48, 0, 0.12),
    ):
        self.client, self.width, self.height = client, width, height
        self.near, self.far = 0.03, 2.5
        self.view = p.computeViewMatrix(eye, target, [0, 0, 1])
        self.projection = p.computeProjectionMatrixFOV(50, width / height, self.near, self.far)
        f = height / (2 * np.tan(np.deg2rad(50) / 2))
        self.k = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
        world_from_gl = np.linalg.inv(np.array(self.view).reshape(4, 4, order="F"))
        self.pose = world_from_gl @ np.diag([1, -1, -1, 1])

    def capture(self, timestamp: float) -> Frame:
        _, _, rgba, buffer, segmentation = p.getCameraImage(
            self.width,
            self.height,
            self.view,
            self.projection,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.client,
        )
        depth_buffer = np.asarray(buffer).reshape(self.height, self.width)
        depth = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer)
        depth[depth_buffer >= 0.9999] = 0
        seg = np.asarray(segmentation, dtype=np.int64).reshape(self.height, self.width)
        ids = np.where(seg < 0, -1, seg & ((1 << 24) - 1))
        return Frame(
            np.asarray(rgba).reshape(self.height, self.width, 4)[..., :3].copy(),
            depth.astype(np.float32),
            ids,
            self.k,
            self.pose,
            timestamp,
        )
