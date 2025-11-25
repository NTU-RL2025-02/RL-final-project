from thrifty.tests.test_cbf import (
    test_cbf_raises_z_when_below_table_limit,
    test_passthrough_without_cvxpy,
)
from thrifty.tests.test_recovery import (
    test_extract_target_xy_prefers_square_nut_pos,
    test_extract_target_xy_requires_xyz_and_returns_xy_only,
    test_get_recovery_action_auto_extracts_target,
    test_airborne_alignment_moves_toward_target_xy_only,
    test_emergency_lift_blocks_xy_motion,
    test_yield_when_within_xy_tolerance,
)
from thrifty.CBF import CBFController


def test_cbf_suite():
    # Aggregate CBF tests to allow running this file alone.
    test_passthrough_without_cvxpy()
    if CBFController().cvxpy_available:
        test_cbf_raises_z_when_below_table_limit()

    # Run recovery expert tests as part of the aggregate suite.
    test_extract_target_xy_prefers_square_nut_pos()
    test_extract_target_xy_requires_xyz_and_returns_xy_only()
    test_get_recovery_action_auto_extracts_target()
    test_emergency_lift_blocks_xy_motion()
    test_airborne_alignment_moves_toward_target_xy_only()
    test_yield_when_within_xy_tolerance()

test_cbf_suite()
