import numpy as np

from thrifty.utils.recovery_plan_near_state import get_recovery_state_from_flat


def test_get_recovery_state_from_flat_extracts_expected_slices():
    # Construct a flat obs matching GymWrapper ordering: eef_pos(3), eef_quat(4),
    # gripper_qpos(2), object(28). Only fill the slices we care about.
    flat = np.zeros(37, dtype=np.float32)

    # eef_pos
    flat[0:3] = [0.1, -0.2, 0.9]

    # object portion starts at idx 9. SquareNut_pos is offsets 7:10 within object.
    object_start = 9
    square_pos_slice = slice(object_start + 7, object_start + 10)
    square_quat_slice = slice(object_start + 10, object_start + 14)
    flat[square_pos_slice] = [0.3, 0.4, 0.83]
    flat[square_quat_slice] = [0.0, 0.0, 0.5, 0.866]

    state = get_recovery_state_from_flat(flat)

    np.testing.assert_array_equal(state["eef_pos"], [0.1, -0.2, 0.9])
    np.testing.assert_array_equal(state["square_pos"], [0.3, 0.4, 0.83])
    np.testing.assert_array_equal(state["square_quat"], [0.0, 0.0, 0.5, 0.866])
