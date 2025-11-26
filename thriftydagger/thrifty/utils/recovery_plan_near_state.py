"""
Thin compatibility layer; the actual helpers now live in
`thrifty.utils.data_driven_recovery`.
"""

from thrifty.utils.data_driven_recovery import (
    get_recovery_state,
    get_recovery_state_from_flat,
)

__all__ = ["get_recovery_state", "get_recovery_state_from_flat"]
