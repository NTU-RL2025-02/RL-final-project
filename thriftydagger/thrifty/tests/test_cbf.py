import numpy as np
import pytest

from thrifty.CBF import CBFController


def test_passthrough_without_cvxpy():
    controller = CBFController()
    controller.cvxpy_available = False  # force fallback path
    obs = {}  # missing robot0_eef_pos
    nominal = np.array([2.0, -2.0, 0.5, 0.1, 0.2, 0.3, -0.4], dtype=np.float32)
    safe = controller.get_safe_action(obs, nominal)
    np.testing.assert_array_equal(safe, np.clip(nominal, -1.0, 1.0))
    print("pass test_passthrough_without_cvxpy")


@pytest.mark.skipif(CBFController().cvxpy_available is False, reason="cvxpy not installed")
def test_cbf_raises_z_when_below_table_limit():
    controller = CBFController(table_z_limit=0.86, action_scale=0.05, alpha=1.0)
    obs = {"robot0_eef_pos": np.array([0.0, 0.0, 0.85], dtype=np.float32)}
    nominal = np.array([0.0, 0.0, -0.5, 0.1, 0.2, 0.3, -0.4], dtype=np.float32)
    safe = controller.get_safe_action(obs, nominal)
    assert safe.shape == nominal.shape
    assert safe[2] >= 0.2 - 1e-4  # CBF constraint should nudge z-velocity upward
    np.testing.assert_array_less(np.abs(safe), np.ones_like(safe) + 1e-6)
    print("pass test_cbf_raises_z_when_below_table_limit")
