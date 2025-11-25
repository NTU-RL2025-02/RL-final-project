import numpy as np
import pytest

from thrifty.Rule_based_recovery import (
    CLEARANCE_Z,
    TOLERANCE_XY,
    DEFAULT_LIFT_SPEED,
    extract_target_xy,
    get_recovery_action,
)


def test_extract_target_xy_prefers_square_nut_pos():
    obs = {
        "SquareNut_pos": np.array([0.2, -0.1, 0.9], dtype=np.float32),
        "object": np.array([0.5, 0.5, 0.5], dtype=np.float32),
    }
    xy = extract_target_xy(obs)
    np.testing.assert_array_equal(xy, np.array([0.2, -0.1], dtype=np.float32))


def test_get_recovery_action_auto_extracts_target():
    obs = {
        "robot0_eef_pos": np.array([0.0, 0.0, CLEARANCE_Z - 0.02], dtype=np.float32),
        "SquareNut_pos": np.array([0.05, 0.01, 0.9], dtype=np.float32),
    }

    action, is_active = get_recovery_action(obs, target_xy=None)

    assert is_active is True
    assert action.shape == (7,)
    np.testing.assert_array_equal(action[:2], np.zeros(2))
    assert action[2] > 0.0


def test_extract_target_xy_requires_xyz_and_returns_xy_only():
    obs = {"SquareNut_pos": np.array([0.3, 0.4, 0.7], dtype=np.float32)}
    xy = extract_target_xy(obs)
    assert xy.shape == (2,)
    np.testing.assert_array_equal(xy, np.array([0.3, 0.4], dtype=np.float32))


def test_emergency_lift_blocks_xy_motion():
    obs = {"robot0_eef_pos": np.array([0.0, 0.0, CLEARANCE_Z - 0.02], dtype=np.float32)}
    target_xy = np.array([0.05, -0.05], dtype=np.float32)

    action, is_active = get_recovery_action(obs, target_xy)

    assert is_active is True
    assert action.shape == (7,)
    np.testing.assert_array_equal(action[:2], np.zeros(2))
    assert action[2] > 0.0
    assert np.isclose(action[2], np.clip(DEFAULT_LIFT_SPEED, 0.0, 1.0))
    np.testing.assert_array_equal(action[3:], np.zeros(4))


def test_airborne_alignment_moves_toward_target_xy_only():
    obs = {"robot0_eef_pos": np.array([0.0, 0.0, CLEARANCE_Z + 0.05], dtype=np.float32)}
    target_xy = np.array([0.03, -0.02], dtype=np.float32)

    action, is_active = get_recovery_action(obs, target_xy)

    assert is_active is True
    assert action.shape == (7,)
    assert action[2] == 0.0
    assert np.sign(action[0]) == np.sign(target_xy[0])
    assert np.sign(action[1]) == np.sign(target_xy[1])
    assert abs(action[0]) > 0
    assert abs(action[1]) > 0
    np.testing.assert_array_equal(action[3:], np.zeros(4))


def test_yield_when_within_xy_tolerance():
    obs = {"robot0_eef_pos": np.array([0.01, -0.01, CLEARANCE_Z + 0.05], dtype=np.float32)}
    target_xy = np.array([0.015, -0.005], dtype=np.float32)

    xy_error = np.linalg.norm(target_xy - obs["robot0_eef_pos"][:2])
    assert xy_error <= TOLERANCE_XY

    action, is_active = get_recovery_action(obs, target_xy)

    assert action is None
    assert is_active is False
