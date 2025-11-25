from thrifty.tests.test_cbf import (
    test_cbf_raises_z_when_below_table_limit,
    test_passthrough_without_cvxpy,
)
from thrifty.CBF import CBFController


def test_cbf_suite():
    # Aggregate CBF tests to allow running this file alone.
    test_passthrough_without_cvxpy()
    if CBFController().cvxpy_available:
        test_cbf_raises_z_when_below_table_limit()

test_cbf_suite()
