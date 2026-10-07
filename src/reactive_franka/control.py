"""cuRobo v2 adapter, deliberately separate from the diagnostic Bullet IK controller."""

import numpy as np
import pybullet as p
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation


def mapping_depth(frame, robot_id, target_mask):
    depth = frame.depth.copy()
    excluded = frame.body_ids == robot_id  # simulation robot self-filter only
    if target_mask is not None:
        excluded |= target_mask
    depth[binary_dilation(excluded, iterations=3)] = 0
    return depth


class BulletIK:
    """Diagnostic only: has no TSDF or collision avoidance."""

    def __init__(self, sim):
        self.sim = sim

    def update_map(self, frame, target_mask):
        pass

    def command(self, position, quat_xyzw):
        q, _ = self.sim.joints()
        goal = p.calculateInverseKinematics(
            self.sim.robot,
            self.sim.hand,
            position,
            quat_xyzw,
            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self.sim.client,
        )[:7]
        return q + np.clip(np.asarray(goal) - q, -0.015, 0.015)


class CuroboMPC:
    def __init__(self, sim, frame, target_mask, dt=1 / 30):
        import torch
        from curobo.model_predictive_control import (
            ModelPredictiveControl,
            ModelPredictiveControlCfg,
        )
        from curobo.perception import Mapper, MapperCfg
        from curobo.scene import Scene
        from curobo.types import DeviceCfg

        if not torch.cuda.is_available():
            raise RuntimeError(
                "cuRobo requires CUDA. Use --controller bullet-ik only for diagnostics."
            )
        self.torch, self.sim = torch, sim
        self.device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
        self.mapper = Mapper(
            MapperCfg(
                voxel_size=0.01,
                esdf_voxel_size=0.02,
                extent_meters_xyz=(1.4, 1.4, 1.2),
                extent_esdf_meters_xyz=(1.4, 1.4, 1.2),
                grid_center=torch.tensor([0.4, 0, 0.4]),
                truncation_distance=0.04,
                depth_minimum_distance=0.05,
                depth_maximum_distance=2.5,
                minimum_tsdf_weight=0.5,
                decay_factor=0.3,
                roughness=3.0,
                num_cameras=1,
                image_height=frame.depth.shape[0],
                image_width=frame.depth.shape[1],
                device="cuda:0",
                block_size=2,
            )
        )
        self.grid = None
        self.update_map(frame, target_mask)
        from importlib.resources import files

        import yaml

        robot = yaml.safe_load(
            files("curobo").joinpath("content/configs/robot/franka.yml").read_text()
        )
        robot["robot_cfg"]["kinematics"]["cspace"]["velocity_scale"] = 0.25
        robot["robot_cfg"]["kinematics"]["cspace"]["max_acceleration"] = 4.0
        cfg = ModelPredictiveControlCfg.create(
            robot=robot,
            scene_model=Scene(voxel=[self.grid]),
            device_cfg=self.device_cfg,
            use_cuda_graph=True,
            optimization_dt=dt,
            interpolation_steps=4,
            optimizer_collision_activation_distance=0.035,
            warm_start_optimization_num_iters=100,
            cold_start_optimization_num_iters=200,
        )
        self.mpc = ModelPredictiveControl(cfg)
        # v2 computes voxel counts through float division, then kernels truncate to int.
        # E.g. 1.2 / 0.02 becomes 59.999996, corrupting the Z stride. Keep exact counts.
        voxel_data = self.mpc.scene_collision_checker.data.voxels
        voxel_data.params[..., :3].copy_(self.tensor(self.grid.feature_tensor.shape))
        self.indices = [self.mpc.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
        if len(self.mpc.joint_names) != 7:
            raise RuntimeError("Expected Franka config with seven active arm joints")
        self.mpc.setup(self.state())
        self.failures = 0

    def tensor(self, data):
        return self.torch.as_tensor(
            np.asarray(data).copy(), device="cuda:0", dtype=self.torch.float32
        )

    def state(self):
        from curobo.types import JointState

        q, dq = self.sim.joints()
        order = [int(name.removeprefix("panda_joint")) - 1 for name in self.mpc.joint_names]
        state = JointState.from_position(
            self.tensor(q[order])[None], joint_names=self.mpc.joint_names
        )
        state.velocity = self.tensor(dq[order])[None]
        state.acceleration = getattr(
            self, "last_acceleration", self.torch.zeros_like(state.position)
        )
        return state

    def update_map(self, frame, target_mask):
        from curobo.types import CameraObservation, Pose

        depth = mapping_depth(frame, self.sim.robot, target_mask)
        quat = Rotation.from_matrix(frame.world_from_camera[:3, :3]).as_quat()
        pose = Pose.from_list(
            [*frame.world_from_camera[:3, 3], quat[3], *quat[:3]], device_cfg=self.device_cfg
        )
        observation = CameraObservation(
            name="bullet",
            depth_image=self.tensor(depth)[None],
            rgb_image=self.torch.as_tensor(frame.rgb.copy(), device="cuda:0")[None],
            intrinsics=self.tensor(frame.intrinsics)[None],
            pose=pose,
            timestamp=frame.timestamp,
        )
        self.mapper.integrate(observation)
        grid = self.mapper.compute_esdf()
        if self.grid is None:
            self.grid = grid
        elif self.grid.feature_tensor.data_ptr() != grid.feature_tensor.data_ptr():
            # Keep the scene tensor address stable for captured CUDA graphs.
            self.grid.feature_tensor.copy_(grid.feature_tensor)

    def command(self, position, quat_xyzw):
        from curobo.types import GoalToolPose, Pose

        state = self.state()  # Bullet position/velocity and ideal-servo acceleration
        poses = self.mpc.compute_kinematics(state).tool_poses.to_dict()
        poses[self.mpc.tool_frames[0]] = Pose.from_list(
            [*position, quat_xyzw[3], *quat_xyzw[:3]], device_cfg=self.device_cfg
        )
        goal = GoalToolPose.from_poses(
            poses, ordered_tool_frames=self.mpc.tool_frames, num_goalset=1
        )
        self.mpc.update_goal_tool_poses(goal, run_ik=False)
        result = self.mpc.optimize_action_sequence(state)
        action = result.action_sequence
        # v2 reports whole-horizon feasibility in `success`; inspect the executed
        # interval itself so an obstacle near the distant goal does not freeze MPC.
        metrics = self.mpc.trajectory_execution_manager.get_current_metrics()
        feasible_steps = metrics.costs_and_constraints.get_feasible(
            include_all_hybrid=False, sum_horizon=False
        )
        feasible = (
            feasible_steps
            if isinstance(feasible_steps, bool)
            else bool(feasible_steps[:, 4:8].all().item())
        )
        if action is None or not feasible or not bool(self.torch.isfinite(action.position).all()):
            self.failures += 1
            q = self.sim.joints()[0]
            self.sim.ideal_trajectory = q[None]
            self.sim.ideal_velocity = np.zeros((1, 7))
            self.last_acceleration = self.torch.zeros_like(state.position)
            self.mpc.reset_robot(state)
            return q
        self.failures = 0
        target = action.position.detach().cpu().numpy().reshape(-1, 7)[-1, self.indices]
        self.last_acceleration = action.acceleration[:, -1, :].clone()
        self.sim.ideal_trajectory = action.position[0, :, self.indices].detach().cpu().numpy()
        self.sim.ideal_velocity = action.velocity[0, :, self.indices].detach().cpu().numpy()
        return target
