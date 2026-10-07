import numpy as np
import pybullet as p

from reactive_franka.geometry import HANDLE_CLOSED_GRIP, HANDLE_GRASP, handle_grasp_pose
from reactive_franka.simulation import HAND_TO_GRASP, STATIONS
from reactive_franka.task import CLEARANCE_HEIGHT, PickPlace
from reactive_franka.tracking import TrackedPose


class Sim:
    def __init__(self):
        self.time = 0.0
        self.hand = np.array([0.4, 0, 0.6])
        self.quat = np.array([1.0, 0, 0, 0])
        self.attached = False
        self.cup, self.client = 1, 0
        self.station = 0

    def hand_pose(self):
        return self.hand.copy(), self.quat.copy()

    def attach(self):
        self.attached = True
        return True

    def release(self):
        self.attached = False

    def resume_target_motion(self, station):
        self.station = station


def observation(now=0):
    return TrackedPose(STATIONS[0].copy(), 0.0, np.zeros(3), 0.0, np.eye(8) * 1e-5, now, now)


def test_missing_observation_waits_within_pick():
    task, sim = PickPlace(), Sim()
    goal, _, grip = task.update(sim, None)
    assert task.phase == "pick" and task.waiting_for_target
    np.testing.assert_allclose(goal, sim.hand)
    assert grip == 0.04
    task.update(sim, observation())
    sim.time = 1
    goal, _, _ = task.update(sim, observation())
    assert task.phase == "pick" and task.step == "preapproach"
    assert task.waiting_for_target
    np.testing.assert_allclose(goal[:2], sim.hand[:2])
    assert goal[2] == CLEARANCE_HEIGHT


def test_gripper_occlusion_does_not_restart_pick():
    task, sim = PickPlace(), Sim()
    estimate = observation()
    task.update(sim, estimate)
    sim.hand = task.goal.copy()
    task.update(sim, estimate)  # Enter approach.
    sim.hand = estimate.position + HANDLE_GRASP + [0, 0, HAND_TO_GRASP + 0.15]
    task.update(sim, estimate)  # Enter descent at the handle.
    sim.time = 0.6
    sim.hand = estimate.position + HANDLE_GRASP + [0, 0, HAND_TO_GRASP]
    task.update(sim, estimate)  # Stale camera frame, but finish the grasp.
    assert task.step == "close"
    sim.time = 1.0
    estimate.timestamp = sim.time
    goal, _, grip = task.update(sim, estimate)
    assert task.phase == "place" and sim.attached
    assert grip == HANDLE_CLOSED_GRIP
    np.testing.assert_allclose(goal, sim.hand)


def test_cycle_counts_only_after_return_home(monkeypatch):
    task, sim = PickPlace(), Sim()
    home = sim.hand.copy()
    sim.quat = np.array([np.cos(0.1), np.sin(0.1), 0, 0])
    home_quaternion = sim.quat.copy()
    monkeypatch.setattr(
        p, "getBasePositionAndOrientation", lambda *a, **kw: (STATIONS[1], sim.quat)
    )
    phases = [task.phase]
    for _ in range(100):
        goal, quat, _ = task.update(sim, observation(sim.time))
        if task.phase != phases[-1]:
            phases.append(task.phase)
        if task.phase == "go_home":
            assert task.cycles == 0
            assert not sim.attached and sim.station == 1
        if task.cycles:
            break
        # Fake perfect waypoint execution; actual MPC is exercised by the demo.
        sim.hand, sim.quat = goal, quat
        sim.time += 0.1
    assert phases == ["pick", "place", "go_home", "pick"]
    assert task.cycles == 1
    np.testing.assert_allclose(sim.hand, home)
    np.testing.assert_allclose(quat, home_quaternion)


def test_descent_tracks_faster_cup_and_long_occlusion_clears_camera_view():
    task, sim = PickPlace(), Sim()
    task.update(sim, observation())
    task.transition("descend", sim.time)
    sim.time = 0.1
    estimate = observation(sim.time)
    estimate.position[0] += 0.03
    estimate.velocity[0] = 0.05
    goal, _, _ = task.update(sim, estimate)
    assert goal[0] > estimate.position[0]  # Camera velocity leads the moving target.
    np.testing.assert_allclose(
        goal[1:], estimate.position[1:] + HANDLE_GRASP[1:] + [0, HAND_TO_GRASP]
    )
    sim.hand = goal.copy()
    sim.time = 2.5  # No new measurement: stop chasing and lift to clear the view.
    goal, _, grip = task.update(sim, estimate)
    assert task.phase == "pick" and task.waiting_for_target
    assert goal[2] == CLEARANCE_HEIGHT and grip == 0.04
    np.testing.assert_allclose(goal[:2], sim.hand[:2])


def test_rotated_handle_grasp_and_place_keep_cup_center_on_station():
    from scipy.spatial.transform import Rotation

    task, sim = PickPlace(), Sim()
    estimate = observation()
    estimate.yaw = np.pi / 2
    task.update(sim, estimate)
    task.transition("descend", sim.time)
    goal, quat, _ = task.update(sim, estimate)
    handle, _ = handle_grasp_pose(estimate.position, estimate.yaw)
    rotation = Rotation.from_quat(quat)
    np.testing.assert_allclose(goal + rotation.apply([0, 0, HAND_TO_GRASP]), handle, atol=1e-9)
    np.testing.assert_allclose(
        handle - estimate.position, [0, HANDLE_GRASP[0], HANDLE_GRASP[2]], atol=1e-9
    )
    np.testing.assert_allclose(rotation.apply([0, 1, 0]), [1, 0, 0], atol=1e-9)
    task.transition("lower", sim.time)
    # A different transport observation must not rotate the attached grasp.
    goal, quat, _ = task.update(sim, observation())
    place_handle, _ = handle_grasp_pose(STATIONS[1], estimate.yaw)
    np.testing.assert_allclose(goal, place_handle + [0, 0, HAND_TO_GRASP + 0.006])
