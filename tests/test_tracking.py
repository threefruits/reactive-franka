import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reactive_franka.perception import Estimate
from reactive_franka.tracking import PoseKalmanFilter, wrap_angle


def observation(timestamp, position, yaw=0.0, rmse=0.005):
    pose = np.eye(4)
    pose[:3, 3] = position
    pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    return Estimate(pose, np.zeros((1, 1), dtype=bool), timestamp, rmse, 0.9)


def test_noisy_camera_updates_reduce_error_and_remain_positive_semidefinite():
    rng = np.random.default_rng(11)
    tracker = PoseKalmanFilter()
    raw_error, filtered_error = [], []
    for timestamp in np.arange(0, 10, 0.1):
        truth = np.array(
            [
                0.52 + 0.10 * np.sin(0.55 * timestamp),
                -0.23,
                0.051,
            ]
        )
        raw = truth + rng.normal(0, 0.008, 3)
        state = tracker.advance(timestamp, observation(timestamp, raw, rmse=0.008))
        raw_error.append(np.linalg.norm(raw - truth))
        filtered_error.append(np.linalg.norm(state.position - truth))
        assert np.linalg.eigvalsh(state.covariance).min() >= -1e-12
    assert np.mean(np.square(filtered_error)) < np.mean(np.square(raw_error))
    assert tracker.updates == 100


def test_predicts_through_occlusion_without_refreshing_camera_age():
    tracker = PoseKalmanFilter()
    for timestamp in np.arange(0, 2.01, 0.1):
        state = tracker.advance(
            timestamp, observation(timestamp, [0.5 + 0.01 * timestamp, 0, 0.05])
        )
    predicted = tracker.advance(2.2)
    assert predicted.position[0] > state.position[0]
    assert predicted.covariance[0, 0] > state.covariance[0, 0]
    assert predicted.observation_timestamp == 2.0
    assert predicted.is_fresh(2.2)
    assert not tracker.advance(3).is_fresh(3)
    recovered = tracker.advance(3.1, observation(3.1, [0.531, 0, 0.05]))
    assert recovered.is_fresh(3.1)
    assert np.linalg.norm(recovered.position - [0.531, 0, 0.05]) < 0.005


def test_every_distinct_measurement_updates_once_and_yaw_wraps():
    tracker = PoseKalmanFilter()
    tracker.advance(0, observation(0, [0.5, 0, 0.05], np.pi - 0.02))
    measurement = observation(0.1, [0.51, 0, 0.06], -np.pi + 0.02)
    state = tracker.advance(0.1, measurement)
    assert 0.5 < state.position[0] < 0.51
    assert abs(wrap_angle(state.yaw - np.pi)) < 0.03
    repeated = tracker.advance(0.1, measurement)
    np.testing.assert_allclose(repeated.covariance, state.covariance)
    assert tracker.updates == 2
    with pytest.raises(ValueError, match="monotonic"):
        tracker.advance(0)
    with pytest.raises(ValueError, match="newer"):
        tracker.advance(0.2, observation(0.3, [0.5, 0, 0.05]))


def test_hidden_handle_updates_position_without_inventing_yaw_measurement():
    tracker = PoseKalmanFilter()
    tracker.advance(0, observation(0, [0.5, 0, 0.05], yaw=0.2))
    hidden = observation(0.1, [0.51, 0, 0.05], yaw=2.5)
    hidden.yaw_observed = False
    state = tracker.advance(0.1, hidden)
    assert 0.5 < state.position[0] < 0.51
    assert abs(state.yaw - 0.2) < 1e-9
    assert tracker.updates == 2
