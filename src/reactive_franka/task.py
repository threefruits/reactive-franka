"""Pick → place → go home; every waypoint is executed by the collision-aware MPC."""

import logging

import numpy as np

from .geometry import HANDLE_CLOSED_GRIP, handle_grasp_pose
from .simulation import HAND_TO_GRASP, STATIONS

LOG = logging.getLogger(__name__)
GRASP_PREDICTION_SECONDS = 2.0
CLEARANCE_HEIGHT = 0.65

PHASES = {
    "preapproach": "pick",
    "approach": "pick",
    "descend": "pick",
    "close": "pick",
    "lift": "place",
    "transfer": "place",
    "lower": "place",
    "open": "place",
    "retreat": "place",
    "home": "go_home",
}


class PickPlace:
    def __init__(self):
        self.phase = "pick"
        self.step = "preapproach"
        self.entered = 0.0
        self.cycles = 0
        self.destination = 1
        self.pick = None
        self.grasp_yaw = 0.0
        self.have_grasp_yaw = False
        self.goal = None
        self.home = None
        self.home_quaternion = None
        self.grip = 0.04
        self.quaternion = np.array([1.0, 0, 0, 0])
        self.waiting_for_target = True

    def transition(self, step, now):
        phase = PHASES[step]
        if phase != self.phase:
            LOG.info("t=%.2f %s -> %s (cycles=%d)", now, self.phase, phase, self.cycles)
        LOG.debug("t=%.2f %s -> %s", now, self.step, step)
        self.phase, self.step, self.entered = phase, step, now

    def update(self, sim, estimate):
        now = sim.time
        hand, hand_quaternion = sim.hand_pose()
        if self.home is None:
            self.home, self.home_quaternion = hand.copy(), hand_quaternion.copy()
            self.goal = hand.copy()
        fresh = estimate is not None and estimate.is_fresh(now)
        # Keep following the Kalman prediction through short gripper occlusions.
        # Limit open-loop pursuit; a longer gap clears the view from above the cup.
        predicted = (
            estimate is not None
            and 0 <= now - estimate.observation_timestamp <= GRASP_PREDICTION_SECONDS
        )
        self.waiting_for_target = False
        if self.phase == "pick":
            if self.step != "preapproach" and not predicted:
                self.transition("preapproach", now)
            if fresh or (self.step != "preapproach" and predicted):
                self.pick = estimate.position.copy()
                if self.step == "preapproach" and np.sqrt(estimate.covariance[3, 3]) < 0.3:
                    self.grasp_yaw = estimate.yaw
                    self.have_grasp_yaw = True
                # A small velocity lead compensates for MPC tracking lag. This uses
                # camera-estimated velocity, never the simulator's motion formula.
                self.pick[:2] += np.clip(estimate.velocity[:2], -0.10, 0.10) * 0.4
            if self.step == "preapproach":
                self.grip = 0.04
                if not fresh or not self.have_grasp_yaw:
                    self.waiting_for_target = True
                    self.goal = hand.copy()
                    if not self.have_grasp_yaw and self.pick is not None:
                        self.goal = self.home.copy()
                    elif self.pick is not None:
                        self.goal[2] = max(hand[2], CLEARANCE_HEIGHT)
                    self.entered = now
                    return self.goal.copy(), self.quaternion, self.grip
        handle, grasp_quaternion = handle_grasp_pose(
            self.pick if self.pick is not None else np.zeros(3), self.grasp_yaw
        )
        place_handle, _ = handle_grasp_pose(STATIONS[self.destination], self.grasp_yaw)
        if self.step != "home":
            self.quaternion = grasp_quaternion
        elapsed = now - self.entered
        if self.step == "preapproach":
            self.goal = np.array([handle[0], handle[1], CLEARANCE_HEIGHT])
            if np.linalg.norm(hand - self.goal) < 0.025:
                self.transition("approach", now)
        elif self.step == "approach":
            self.grip = 0.04
            self.goal = handle + [0, 0, HAND_TO_GRASP + 0.15]
            if np.linalg.norm(hand - self.goal) < 0.02:
                self.transition("descend", now)
        elif self.step == "descend":
            self.goal = handle + [0, 0, HAND_TO_GRASP]
            if np.linalg.norm(hand - self.goal) < 0.020:
                self.transition("close", now)
        elif self.step == "close":
            self.goal = handle + [0, 0, HAND_TO_GRASP]
            self.grip = HANDLE_CLOSED_GRIP
            if elapsed > 0.3 and sim.attach():
                self.transition("lift", now)
            elif elapsed > 2.0:
                # A missed grasp stays in pick: reopen and reacquire above the cup.
                self.transition("preapproach", now)
                self.grip = 0.04
        elif self.step == "lift":
            self.goal = np.array([handle[0], handle[1], CLEARANCE_HEIGHT])
            if np.linalg.norm(hand - self.goal) < 0.025:
                self.transition("transfer", now)
        elif self.step == "transfer":
            self.goal = np.array([*place_handle[:2], CLEARANCE_HEIGHT])
            if np.linalg.norm(hand - self.goal) < 0.025:
                self.transition("lower", now)
        elif self.step == "lower":
            self.goal = place_handle + [0, 0, HAND_TO_GRASP + 0.006]
            if np.linalg.norm(hand - self.goal) < 0.012:
                sim.release()
                self.transition("open", now)
        elif self.step == "open":
            self.grip = 0.04
            if elapsed > 0.5:
                self.transition("retreat", now)
        elif self.step == "retreat":
            self.goal = np.array([*place_handle[:2], CLEARANCE_HEIGHT])
            if np.linalg.norm(hand - self.goal) < 0.025:
                # Validate release before restarting the kinematic platform motion.
                import pybullet as p

                placed = np.array(
                    p.getBasePositionAndOrientation(sim.cup, physicsClientId=sim.client)[0]
                )
                if np.linalg.norm(placed - STATIONS[self.destination]) > 0.045:
                    raise RuntimeError(f"Placement missed station: {placed}")
                sim.resume_target_motion(self.destination)
                self.destination = 1 - self.destination
                self.transition("home", now)
        elif self.step == "home":
            self.goal = self.home.copy()
            self.quaternion = self.home_quaternion.copy()
            if np.linalg.norm(hand - self.home) < 0.02 and abs(
                np.dot(hand_quaternion, self.home_quaternion)
            ) > np.cos(0.05 / 2):
                self.cycles += 1
                self.pick = None
                self.have_grasp_yaw = False
                self.transition("preapproach", now)
        if now - self.entered > 25:
            raise RuntimeError(
                f"Task stuck in {self.phase}/{self.step}; inspect perception and MPC diagnostics"
            )
        return self.goal.copy(), self.quaternion, self.grip
