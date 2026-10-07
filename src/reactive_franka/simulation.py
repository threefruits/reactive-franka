"""Small deterministic scene. Object motion is kinematic until grasped."""

from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from .camera import Camera
from .geometry import HANDLE_GRASP, cup_collision, cup_mesh

HOME = [0.0, -0.65, 0.0, -2.25, 0.0, 1.65, 0.78]
STATIONS = (np.array([0.52, -0.23, 0.051]), np.array([0.52, 0.23, 0.051]))
HAND_TO_GRASP = 0.105
CUP_TRAVEL = 0.10  # meters either side of the placement point
CUP_ANGULAR_SPEED = 0.55  # rad/s; peak speed 5.5 cm/s, matching the red obstacle


class Simulation:
    def __init__(self, gui: bool, output: Path):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1 / 240, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=self.client)
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.client
        )
        names = {
            p.getJointInfo(self.robot, i, physicsClientId=self.client)[1].decode(): i
            for i in range(p.getNumJoints(self.robot, physicsClientId=self.client))
        }
        self.arm = [names[f"panda_joint{i}"] for i in range(1, 8)]
        self.fingers = [names["panda_finger_joint1"], names["panda_finger_joint2"]]
        links = {
            p.getJointInfo(self.robot, i, physicsClientId=self.client)[12].decode(): i
            for i in range(p.getNumJoints(self.robot, physicsClientId=self.client))
        }
        self.hand = links["panda_hand"]
        for j, q in zip(self.arm + self.fingers, HOME + [0.04, 0.04]):
            p.resetJointState(self.robot, j, q, physicsClientId=self.client)
        self.table = self.box([0.48, 0, -0.04], [0.43, 0.48, 0.04], [0.68, 0.58, 0.45, 1])
        self.obstacle = self.box([0.52, 0, 0.16], [0.035, 0.055, 0.16], [0.85, 0.15, 0.12, 1])
        self.air_obstacles = (self.box([0.62, 0, 0.43], [0.035, 0.04, 0.04], [0.15, 0.4, 0.9, 1]),)
        self.obstacles = (self.obstacle, *self.air_obstacles)
        output.mkdir(parents=True, exist_ok=True)
        mesh_path = output / "cup.obj"
        cup_mesh().export(mesh_path)
        visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(mesh_path.resolve()),
            rgbaColor=[0.85, 0.92, 1, 1],
            physicsClientId=self.client,
        )
        collision = cup_collision(self.client)
        self.cup = p.createMultiBody(0, collision, visual, STATIONS[0], physicsClientId=self.client)
        self.camera = Camera(self.client)
        self.time = 0.0
        self.station = 0
        self.motion_origin = STATIONS[0].copy()
        self.motion_started = 0.0
        self.motion_orientation = [0, 0, 0, 1]
        self.moving = True
        self.constraint = None
        self.command = np.array(HOME)
        self.grip = 0.04
        p.resetDebugVisualizerCamera(1.5, 45, -35, [0.4, 0, 0.2], physicsClientId=self.client)

    def box(self, position, half, color):
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        v = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.client
        )
        return p.createMultiBody(0, c, v, position, physicsClientId=self.client)

    def joints(self):
        states = p.getJointStates(self.robot, self.arm, physicsClientId=self.client)
        return np.array([s[0] for s in states]), np.array([s[1] for s in states])

    def hand_pose(self):
        state = p.getLinkState(
            self.robot, self.hand, computeForwardKinematics=True, physicsClientId=self.client
        )
        return np.array(state[4]), np.array(state[5])

    def grasp_position(self):
        pos, quat = self.hand_pose()
        return pos + np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3) @ [0, 0, HAND_TO_GRASP]

    def move_obstacles(self, time: float):
        """Move the table obstacle and nearby blue obstacle along straight-line paths."""
        positions = (
            [0.52 + 0.10 * np.sin(0.55 * time), 0, 0.16],
            [0.62, 0.12 * np.sin(0.45 * time), 0.43],
        )
        for body, position in zip(self.obstacles, positions):
            p.resetBasePositionAndOrientation(
                body, position, [0, 0, 0, 1], physicsClientId=self.client
            )

    def step(self, command, grip: float, ticks: int = 8):
        self.command = np.asarray(command)
        self.grip = grip
        for tick in range(ticks):
            self.time += 1 / 240
            if self.moving:
                offset = CUP_TRAVEL * np.sin(CUP_ANGULAR_SPEED * (self.time - self.motion_started))
                p.resetBasePositionAndOrientation(
                    self.cup,
                    self.motion_origin + [offset, 0, 0],
                    self.motion_orientation,
                    physicsClientId=self.client,
                )
            self.move_obstacles(self.time)
            p.setJointMotorControlArray(
                self.robot,
                self.arm,
                p.POSITION_CONTROL,
                targetPositions=self.command,
                targetVelocities=[0.0] * 7,
                forces=[87] * 4 + [12] * 3,
                positionGains=[0.25] * 7,
                physicsClientId=self.client,
            )
            p.setJointMotorControlArray(
                self.robot,
                self.fingers,
                p.POSITION_CONTROL,
                targetPositions=[grip, grip],
                forces=[30, 30],
                physicsClientId=self.client,
            )
            # MPC uses ideal joint actuation; the diagnostic IK mode uses the motors.
            # Object contacts, gripper closure, and release still use Bullet dynamics.
            if hasattr(self, "ideal_trajectory"):
                index = min(
                    tick * len(self.ideal_trajectory) // ticks, len(self.ideal_trajectory) - 1
                )
                for j, value, velocity in zip(
                    self.arm, self.ideal_trajectory[index], self.ideal_velocity[index]
                ):
                    p.resetJointState(self.robot, j, value, velocity, physicsClientId=self.client)
            p.stepSimulation(physicsClientId=self.client)
        if hasattr(self, "ideal_trajectory"):
            for j, value, velocity in zip(
                self.arm, self.ideal_trajectory[-1], self.ideal_velocity[-1]
            ):
                p.resetJointState(self.robot, j, value, velocity, physicsClientId=self.client)
            p.performCollisionDetection(physicsClientId=self.client)

    def attach(self) -> bool:
        position, orientation = p.getBasePositionAndOrientation(
            self.cup, physicsClientId=self.client
        )
        cup_rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        error = cup_rotation.T @ (self.grasp_position() - position) - HANDLE_GRASP
        if np.any(np.abs(error) > [0.012, 0.010, 0.012]):
            return False
        hand_pos, hand_quat = self.hand_pose()
        hand_rotation = np.array(p.getMatrixFromQuaternion(hand_quat)).reshape(3, 3)
        relative = cup_rotation.T @ hand_rotation
        if -relative[2, 2] < np.cos(np.deg2rad(15)) or abs(relative[1, 1]) < np.cos(np.deg2rad(20)):
            return False
        if (
            max(
                p.getJointState(self.robot, j, physicsClientId=self.client)[0] for j in self.fingers
            )
            > 0.010
        ):
            return False
        inv = p.invertTransform(hand_pos, hand_quat)
        rel_pos, rel_quat = p.multiplyTransforms(*inv, position, orientation)
        self.moving = False
        p.changeDynamics(self.cup, -1, mass=0.08, physicsClientId=self.client)
        self.constraint = p.createConstraint(
            self.robot,
            self.hand,
            self.cup,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            rel_pos,
            [0, 0, 0],
            parentFrameOrientation=rel_quat,
            physicsClientId=self.client,
        )
        p.changeConstraint(self.constraint, maxForce=100, physicsClientId=self.client)
        # Contact is represented by the grasp constraint during transport.
        for link in [self.hand, *self.fingers]:
            p.setCollisionFilterPair(self.robot, self.cup, link, -1, 0, physicsClientId=self.client)
        return True

    def release(self):
        if self.constraint is not None:
            p.removeConstraint(self.constraint, physicsClientId=self.client)
            self.constraint = None

    def resume_target_motion(self, station: int):
        self.station = station
        # Restart from the released position without jumping to a distant path phase.
        position, orientation = p.getBasePositionAndOrientation(
            self.cup, physicsClientId=self.client
        )
        self.motion_origin = np.array(position)
        self.motion_orientation = p.getQuaternionFromEuler(
            [0, 0, p.getEulerFromQuaternion(orientation)[2]]
        )
        self.motion_started = self.time
        p.changeDynamics(self.cup, -1, mass=0, physicsClientId=self.client)
        for link in [self.hand, *self.fingers]:
            p.setCollisionFilterPair(self.robot, self.cup, link, -1, 1, physicsClientId=self.client)
        self.moving = True

    def collision_count(self):
        return sum(
            c[8] < -0.003
            for obstacle in self.obstacles
            for c in p.getContactPoints(self.robot, obstacle, physicsClientId=self.client)
        )

    def close(self):
        p.disconnect(self.client)
