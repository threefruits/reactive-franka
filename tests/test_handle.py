import numpy as np
import pybullet as p

from reactive_franka.geometry import HANDLE_CLOSED_GRIP, HANDLE_GRASP, handle_grasp_pose
from reactive_franka.simulation import HAND_TO_GRASP, STATIONS, Simulation


def test_handle_collision_has_outer_arc_and_open_hole(tmp_path):
    sim = Simulation(False, tmp_path)
    try:
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001, physicsClientId=sim.client)
        probe = p.createMultiBody(0, shape, physicsClientId=sim.client)
        for offset, contact in [(HANDLE_GRASP, True), ([0.043, 0, 0.005], False)]:
            p.resetBasePositionAndOrientation(
                probe, STATIONS[0] + offset, [0, 0, 0, 1], physicsClientId=sim.client
            )
            assert (
                bool(p.getClosestPoints(sim.cup, probe, 0, physicsClientId=sim.client)) == contact
            )
    finally:
        sim.close()


def test_only_closed_aligned_handle_grasp_can_attach(tmp_path):
    sim = Simulation(False, tmp_path)
    try:
        point, quat = handle_grasp_pose(STATIONS[0], 0)

        def set_gripper(grasp_point, orientation, opening):
            target = grasp_point + [0, 0, HAND_TO_GRASP]
            for _ in range(3):
                joints = p.calculateInverseKinematics(
                    sim.robot,
                    sim.hand,
                    target,
                    orientation,
                    maxNumIterations=500,
                    residualThreshold=1e-7,
                    physicsClientId=sim.client,
                )
                for joint, value in zip(sim.arm, joints):
                    p.resetJointState(sim.robot, joint, value, physicsClientId=sim.client)
            for joint in sim.fingers:
                p.resetJointState(sim.robot, joint, opening, physicsClientId=sim.client)
            np.testing.assert_allclose(sim.grasp_position(), grasp_point, atol=0.001)

        set_gripper(STATIONS[0], quat, HANDLE_CLOSED_GRIP)
        assert not sim.attach()  # The old body-center grasp is no longer accepted.
        set_gripper(point, quat, 0.04)
        assert not sim.attach()  # An open gripper cannot latch the handle.
        _, wrong_yaw = handle_grasp_pose(STATIONS[0], np.pi / 2)
        set_gripper(point, wrong_yaw, HANDLE_CLOSED_GRIP)
        assert not sim.attach()
        set_gripper(point, quat, HANDLE_CLOSED_GRIP)
        assert sim.attach()
        assert sim.constraint is not None and not sim.moving
    finally:
        sim.close()


def test_fixed_camera_sees_cup_during_handle_grasp_at_both_stations(tmp_path):
    from reactive_franka.perception import Perception

    sim = Simulation(False, tmp_path)
    try:
        extrinsics = sim.camera.pose.copy()
        for station in STATIONS:
            for x in [0.42, 0.52, 0.62]:
                center = np.array([x, station[1], station[2]])
                p.resetBasePositionAndOrientation(
                    sim.cup, center, [0, 0, 0, 1], physicsClientId=sim.client
                )
                handle, orientation = handle_grasp_pose(center, 0)
                for _ in range(3):
                    joints = p.calculateInverseKinematics(
                        sim.robot,
                        sim.hand,
                        handle + [0, 0, HAND_TO_GRASP],
                        orientation,
                        maxNumIterations=300,
                        physicsClientId=sim.client,
                    )
                    for joint, value in zip(sim.arm, joints):
                        p.resetJointState(sim.robot, joint, value, physicsClientId=sim.client)
                frame = sim.camera.capture(0)
                assert (frame.body_ids == sim.cup).sum() > 150
                estimate = Perception("sim-mask", sim.cup, "").estimate(frame)
                assert estimate is not None
                assert np.linalg.norm(estimate.position - center) < 0.015
                np.testing.assert_array_equal(frame.world_from_camera, extrinsics)
    finally:
        sim.close()
