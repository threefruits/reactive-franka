"""Shared metric cup geometry, used both for rendering and ICP (no pose labels)."""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

CUP_RADIUS = 0.03
CUP_HEIGHT = 0.10
HANDLE_CENTER = np.array([0.036, 0.0, 0.005])
HANDLE_RADIUS = 0.022
HANDLE_TUBE_RADIUS = 0.005
# Pinch the outer arc across its thickness, clear of the cup wall.
HANDLE_GRASP = HANDLE_CENTER + [HANDLE_RADIUS, 0, 0]
HANDLE_CLOSED_GRIP = 0.002


def handle_grasp_pose(position, yaw):
    rotation = Rotation.from_euler("z", yaw)
    point = np.asarray(position) + rotation.apply(HANDLE_GRASP)
    # Local Z points down; the fingers close along local Y, across the handle plane.
    orientation = (rotation * Rotation.from_euler("x", np.pi)).as_quat()
    return point, orientation


def cup_collision(client):
    """Solid body proxy plus a compound capsule ring matching the visible handle."""
    import pybullet as p

    count = 12
    angles = (np.arange(count) + 0.5) * 2 * np.pi / count
    radius = HANDLE_RADIUS * np.cos(np.pi / count)
    centers = HANDLE_CENTER + radius * np.column_stack(
        [np.cos(angles), np.zeros(count), np.sin(angles)]
    )
    return p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_CYLINDER] + [p.GEOM_CAPSULE] * count,
        radii=[CUP_RADIUS] + [HANDLE_TUBE_RADIUS] * count,
        lengths=[CUP_HEIGHT] + [2 * HANDLE_RADIUS * np.sin(np.pi / count)] * count,
        collisionFramePositions=[[0, 0, 0], *centers.tolist()],
        collisionFrameOrientations=[
            [0, 0, 0, 1],
            *[p.getQuaternionFromEuler([0, -angle, 0]) for angle in angles],
        ],
        physicsClientId=client,
    )


def cup_mesh() -> trimesh.Trimesh:
    # A hollow ceramic cup, centered at its body's midpoint, plus a small handle.
    n = 64
    vertices = []
    for radius, z in [(0.03, -0.05), (0.03, 0.05), (0.025, 0.05), (0.025, -0.044)]:
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
            vertices.append([radius * np.cos(a), radius * np.sin(a), z])
    faces = []
    for ring in range(3):
        for i in range(n):
            a, b = ring * n + i, ring * n + (i + 1) % n
            faces.extend([[a, b, b + n], [a, b + n, a + n]])
    # Bottom surfaces.
    vertices.extend([[0, 0, -0.05], [0, 0, -0.044]])
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[4 * n, j, i], [4 * n + 1, 3 * n + i, 3 * n + j]])
    body = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    handle = trimesh.creation.torus(major_radius=HANDLE_RADIUS, minor_radius=HANDLE_TUBE_RADIUS)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    handle.apply_translation(HANDLE_CENTER)
    return trimesh.util.concatenate([body, handle])


def model_points(count: int = 2500) -> np.ndarray:
    return trimesh.sample.sample_surface(cup_mesh(), count, seed=7)[0]
