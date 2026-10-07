"""Opt-in CUDA integration regression; RUN_GPU_TESTS=1 uv run --extra gpu pytest."""

import os

import numpy as np
import pybullet as p
import pytest

from reactive_franka.control import CuroboMPC
from reactive_franka.perception import Perception
from reactive_franka.simulation import Simulation


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="opt-in CUDA test")
def test_voxel_stride_fk_and_live_updates(tmp_path):
    import torch
    from curobo.types import RobotState

    sim = Simulation(False, tmp_path)
    try:
        frame = sim.camera.capture(0)
        perception = Perception("sim-mask", sim.cup, "")
        estimate = perception.estimate(frame)
        controller = CuroboMPC(sim, frame, estimate.mask)
        voxels = controller.mpc.scene_collision_checker.data.voxels
        expected = torch.tensor(controller.grid.feature_tensor.shape, device="cuda:0")
        torch.testing.assert_close(voxels.params[0, 0, :3], expected.float())
        # Float-to-int truncation previously corrupted the Z stride (60 -> 59).
        assert tuple(voxels.params[0, 0, :3].int().tolist()) == (70, 70, 60)
        assert voxels.features.data_ptr() == controller.grid.feature_tensor.data_ptr()

        target = [0.54, -0.23, 0.31]
        joints = p.calculateInverseKinematics(
            sim.robot,
            sim.hand,
            target,
            [1, 0, 0, 0],
            maxNumIterations=200,
            physicsClientId=sim.client,
        )
        for index, value in zip(sim.arm, joints):
            p.resetJointState(sim.robot, index, value, physicsClientId=sim.client)
        state = controller.state()
        kin = controller.mpc.compute_kinematics(state)
        np.testing.assert_allclose(
            kin.tool_poses.position.cpu().numpy().reshape(3), sim.hand_pose()[0], atol=1e-5
        )
        state = state.unsqueeze(1)
        state.dt = controller.tensor([1 / 30])
        metrics = controller.mpc.metrics_rollout.compute_metrics_from_state(
            RobotState(joint_state=state, cuda_robot_model_state=kin)
        )
        constraints = metrics.costs_and_constraints.constraints
        scene_cost = constraints.values[constraints.names.index("scene_collision")]
        assert scene_cost.max().item() == 0  # known clear pose, no phantom voxel collision

        before = controller.grid.feature_tensor.clone()
        p.resetBasePositionAndOrientation(
            sim.obstacle, [0.7, 0, 0.16], [0, 0, 0, 1], physicsClientId=sim.client
        )
        for index in range(6):
            frame = sim.camera.capture(index / 10)
            estimate = perception.estimate(frame)
            controller.update_map(frame, estimate.mask if estimate is not None else None)
        assert torch.count_nonzero(before != controller.grid.feature_tensor).item() > 100
        assert voxels.features.data_ptr() == controller.grid.feature_tensor.data_ptr()
    finally:
        sim.close()
