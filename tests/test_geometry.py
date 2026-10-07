import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from reactive_franka.control import mapping_depth
from reactive_franka.geometry import model_points
from reactive_franka.perception import Perception, upright_icp
from reactive_franka.simulation import STATIONS, Simulation


def test_depth_camera_and_icp(tmp_path):
    sim = Simulation(False, tmp_path)
    try:
        frame = sim.camera.capture(0)
        mask = frame.body_ids == sim.cup
        points = frame.points(mask)
        assert len(points) > 100
        assert np.linalg.norm(np.median(points, axis=0)[:2] - STATIONS[0][:2]) < 0.04
        assert 0.085 < points[:, 2].max() < 0.11
        estimate = Perception("sim-mask", sim.cup, "").estimate(frame)
        assert estimate is not None
        assert np.linalg.norm(estimate.position - STATIONS[0]) < 0.015
        depth = mapping_depth(frame, sim.robot, mask)
        assert np.all(depth[mask] == 0)
        assert np.all(depth[frame.body_ids == sim.robot] == 0)
        for obstacle in sim.obstacles:
            assert np.any(depth[frame.body_ids == obstacle] > 0)
    finally:
        sim.close()


def test_icp_recovers_known_pose_with_outliers():
    model = model_points()
    rotation = Rotation.from_euler("z", 0.25).as_matrix()
    translation = np.array([0.5, -0.2, 0.051])
    observed = model @ rotation.T + translation
    observed = np.concatenate([observed, [[2, 2, 2]] * 30])
    initial = np.eye(4)
    initial[:3, 3] = translation + [0.003, -0.003, 0.002]
    fitted, rmse, fraction = upright_icp(model, observed, initial, iterations=80)
    assert np.linalg.norm(fitted[:3, 3] - translation) < 0.002
    assert Rotation.from_matrix(fitted[:3, :3] @ rotation.T).magnitude() < 0.05
    assert rmse < 0.002
    assert fraction > 0.98


def test_fixed_camera_estimates_lifted_cup(tmp_path):
    sim = Simulation(False, tmp_path)
    try:
        extrinsics = sim.camera.pose.copy()
        perception = Perception("sim-mask", sim.cup, "")
        assert perception.estimate(sim.camera.capture(0)) is not None
        lifted = [0.52, -0.23, 0.30]
        p.resetBasePositionAndOrientation(sim.cup, lifted, [0, 0, 0, 1], physicsClientId=sim.client)
        frame = sim.camera.capture(0.1)
        estimate = perception.estimate(frame)
        assert estimate is not None
        assert np.linalg.norm(estimate.position - lifted) < 0.02
        np.testing.assert_array_equal(frame.world_from_camera, extrinsics)
        assert len(sim.air_obstacles) == 1
        assert (
            p.getBasePositionAndOrientation(sim.air_obstacles[0], physicsClientId=sim.client)[0][0]
            == 0.62
        )
    finally:
        sim.close()


def test_detection_association_does_not_switch_to_blue_obstacle(tmp_path):
    from types import SimpleNamespace

    class Mask:
        def __init__(self, array):
            self.array = array

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    sim = Simulation(False, tmp_path)
    try:
        frame = sim.camera.capture(0)
        perception = Perception("sim-mask", sim.cup, "")
        assert perception.estimate(frame) is not None
        cup_mask = frame.body_ids == sim.cup
        # Simulate a false positive with higher confidence than the real target.
        result = SimpleNamespace(
            names={0: "cup"},
            boxes=SimpleNamespace(cls=np.array([0, 0]), conf=np.array([0.99, 0.8])),
            masks=SimpleNamespace(
                data=[Mask(frame.body_ids == sim.air_obstacles[0]), Mask(cup_mask)]
            ),
        )
        perception.mode = "yolo"
        perception.detector = SimpleNamespace(predict=lambda *args, **kwargs: [result])
        np.testing.assert_array_equal(perception.detect_mask(sim.camera.capture(0.1)), cup_mask)
    finally:
        sim.close()
