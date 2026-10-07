"""World-frame Kalman tracking from the fixed RGB-D camera's upright ICP poses."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .perception import Estimate


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class TrackedPose:
    position: np.ndarray
    yaw: float
    velocity: np.ndarray
    yaw_rate: float
    covariance: np.ndarray
    timestamp: float
    observation_timestamp: float

    def is_fresh(self, now: float) -> bool:
        # A prediction advances state time, never the last camera observation time.
        return (
            0 <= now - self.observation_timestamp <= 0.8
            and np.sqrt(np.diag(self.covariance)[:3]).max() <= 0.04
        )

    @property
    def world_from_object(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("z", self.yaw).as_matrix()
        pose[:3, 3] = self.position
        return pose


class PoseKalmanFilter:
    """Constant velocity state [x, y, z, yaw, vx, vy, vz, yaw_rate].

    Each distinct valid ICP observation performs one measurement update, including
    during transport. No simulator object pose, grasp state, or prescribed motion model
    enters the filter. Process noise allows changes in velocity during pick/place.
    """

    def __init__(self):
        self.state = None
        self.covariance = None
        self.timestamp = None
        self.observation_timestamp = None
        self.updates = 0
        self.acceleration_density = np.square([0.2, 0.2, 0.2, 0.15])

    def _predict(self, timestamp):
        if self.state is None:
            return
        dt = timestamp - self.timestamp
        if dt < -1e-9:
            raise ValueError("Kalman timestamps must be monotonic")
        dt = max(0.0, dt)
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4) * dt
        spectral_density = np.diag(self.acceleration_density)
        noise = np.block(
            [
                [spectral_density * dt**3 / 3, spectral_density * dt**2 / 2],
                [spectral_density * dt**2 / 2, spectral_density * dt],
            ]
        )
        self.state = transition @ self.state
        self.state[3] = wrap_angle(self.state[3])
        self.covariance = transition @ self.covariance @ transition.T + noise
        self.timestamp = timestamp

    def advance(self, timestamp: float, observation: Estimate | None = None):
        if observation is not None:
            if observation.timestamp > timestamp + 1e-9:
                raise ValueError("Observation cannot be newer than the prediction time")
            # Do not count a reused image as independent evidence.
            if (
                self.observation_timestamp is None
                or observation.timestamp > self.observation_timestamp
            ):
                self._predict(observation.timestamp)
                yaw = Rotation.from_matrix(observation.world_from_object[:3, :3]).as_euler("xyz")[2]
                measurement = np.r_[observation.position, yaw]
                if not np.isfinite(measurement).all():
                    raise ValueError("Kalman measurement must be finite")
                # ICP residuals include systematic depth/model error. Keep a 3 mm floor;
                # uncertain handle orientation gets a much larger angular variance.
                position_std = max(0.003, observation.rmse) / np.sqrt(
                    max(observation.inlier_fraction, 0.1)
                )
                noise = np.diag([position_std**2] * 3 + [0.15**2])
                dimensions = 4 if observation.yaw_observed else 3
                if self.state is None:
                    self.state = np.r_[measurement, np.zeros(4)]
                    self.covariance = np.zeros((8, 8))
                    self.covariance[:4, :4] = noise
                    if not observation.yaw_observed:
                        self.covariance[3, 3] = np.pi**2
                    self.covariance[4:, 4:] = np.diag([0.1**2] * 3 + [0.6**2])
                else:
                    innovation = measurement[:dimensions] - self.state[:dimensions]
                    if observation.yaw_observed:
                        innovation[3] = wrap_angle(innovation[3])
                    noise = noise[:dimensions, :dimensions]
                    gain = np.linalg.solve(
                        self.covariance[:dimensions, :dimensions] + noise,
                        self.covariance[:dimensions, :],
                    ).T
                    self.state += gain @ innovation
                    self.state[3] = wrap_angle(self.state[3])
                    residual = np.eye(8)
                    residual[:, :dimensions] -= gain
                    # Joseph form preserves positive semidefiniteness through long runs.
                    self.covariance = (
                        residual @ self.covariance @ residual.T + gain @ noise @ gain.T
                    )
                    self.covariance = (self.covariance + self.covariance.T) / 2
                self.timestamp = self.observation_timestamp = observation.timestamp
                self.updates += 1
        self._predict(timestamp)
        if self.state is None:
            return None
        return TrackedPose(
            self.state[:3].copy(),
            float(self.state[3]),
            self.state[4:7].copy(),
            float(self.state[7]),
            self.covariance.copy(),
            self.timestamp,
            self.observation_timestamp,
        )
