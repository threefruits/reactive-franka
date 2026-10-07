import numpy as np
import pybullet as p

from reactive_franka.simulation import HOME, STATIONS, Simulation


def test_cup_moves_along_line_at_obstacle_speed_and_resumes_without_jump(tmp_path):
    sim = Simulation(False, tmp_path)
    try:
        positions = []
        for _ in range(360):
            sim.step(HOME, 0.04)
            positions.append(p.getBasePositionAndOrientation(sim.cup, sim.client)[0])
        positions = np.array(positions)
        np.testing.assert_allclose(positions[:, 1], STATIONS[0][1], atol=1e-6)
        np.testing.assert_allclose(positions[:, 2], STATIONS[0][2], atol=1e-6)
        assert 0.199 < np.ptp(positions[:, 0]) <= 0.201  # 20 cm full travel span
        velocity = np.diff(positions[:, 0]) * 30
        assert 0.054 < velocity.max() < 0.056
        assert -0.056 < velocity.min() < -0.054

        released_position = STATIONS[1] + [-0.012, 0.006, 0]
        p.resetBasePositionAndOrientation(
            sim.cup, released_position, [0, 0, 0, 1], physicsClientId=sim.client
        )
        sim.resume_target_motion(1)
        sim.step(HOME, 0.04, ticks=1)
        restarted = np.array(p.getBasePositionAndOrientation(sim.cup, sim.client)[0])
        assert np.linalg.norm(restarted - released_position) < 0.001
        assert restarted[0] > released_position[0]
    finally:
        sim.close()
