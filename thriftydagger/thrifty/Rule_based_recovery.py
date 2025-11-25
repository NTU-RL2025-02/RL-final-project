import numpy as np

# Tunable safety bands for the micro-retreat logic
CLEARANCE_Z = 0.1  # safe hover height in meters
TOLERANCE_XY = 0.015  # allowable XY misalignment in meters
DEFAULT_LIFT_SPEED = 0.5  # upward command used during emergency lift
DEFAULT_XY_GAIN = 5.0  # proportional gain for XY correction
MAX_ACTION = 1.0  # robosuite actions are typically clipped to [-1, 1]


def extract_target_xy(obs):
    """
    Extract XY goal location for alignment from a robosuite observation dict.

    Preferred key for NutAssemblySquare: ``SquareNut_pos`` (x, y, z).
    Falls back to common robosuite object keys if the preferred key is missing.
    """
    def pick(keys, require_three=False):
        for k in keys:
            if k in obs:
                arr = np.asarray(obs[k], dtype=np.float32).reshape(-1)
                if require_three and arr.shape[0] < 3:
                    raise ValueError(f"Expected {k} to have at least 3 elements (x, y, z); got shape {arr.shape}")
                if arr.shape[0] >= 2:
                    # Explicitly split XY from XYZ-style input to avoid passing Z through.
                    return np.array([arr[0], arr[1]], dtype=np.float32)
        return None

    # SquareNut_pos is known to be (x, y, z); require_three enforces the expected shape.
    xy = pick(["SquareNut_pos"], require_three=True)
    if xy is None:
        xy = pick(["peg_pos", "nut_pos", "object", "object-state"])
    if xy is None:
        raise KeyError(
            f"Could not find target XY in obs keys {list(obs.keys())}; "
            "expected one of ['SquareNut_pos', 'peg_pos', 'nut_pos', 'object', 'object-state']"
        )
    return xy


def get_recovery_action(
    obs,
    target_xy=None,
    kp_xy: float = DEFAULT_XY_GAIN,
    clearance_z: float = CLEARANCE_Z,
    tolerance_xy: float = TOLERANCE_XY,
    lift_speed: float = DEFAULT_LIFT_SPEED,
    max_action: float = MAX_ACTION,
):
    """
    Rule-based recovery expert implementing a micro-retreat strategy.

    Args:
        obs: observation dictionary that must contain ``robot0_eef_pos`` (x, y, z).
        target_xy: iterable with the desired XY position for alignment. If None,
            the function will call ``extract_target_xy`` using the provided obs.
        kp_xy: proportional gain used for XY alignment when safely above the part.
        clearance_z: minimum z height considered collision-safe.
        tolerance_xy: maximum XY error allowed before triggering recovery.
        lift_speed: upward command applied during the emergency lift branch.
        max_action: absolute bound for each action dimension (consistent with robosuite).

    Returns:
        action: np.ndarray shaped (7,) or None when yielding to the policy.
        is_active: bool flag indicating whether recovery logic overrides the policy.
    """
    if obs is None or "robot0_eef_pos" not in obs:
        raise ValueError("Observation dict must contain key 'robot0_eef_pos'.")

    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    if eef_pos.shape[0] < 3:
        raise ValueError("Expected robot0_eef_pos to have at least 3 elements (x, y, z).")

    if target_xy is None:
        target_xy = extract_target_xy(obs)
    else:
        target_xy = np.asarray(target_xy, dtype=np.float32).reshape(-1)
        if target_xy.shape[0] != 2:
            raise ValueError("target_xy must be length 2 representing desired (x, y).")

    current_xy = eef_pos[:2]
    current_z = float(eef_pos[2])

    xy_error_vec = target_xy - current_xy
    xy_error = float(np.linalg.norm(xy_error_vec))

    # Case C: already aligned; yield to the learned policy.
    if xy_error <= tolerance_xy:
        return None, False

    # Initialize full action vector; orientation + gripper stay untouched (zeros).
    action = np.zeros(7, dtype=np.float32)

    # Case A: emergency lift when too close to the hardware and misaligned.
    if current_z < clearance_z:
        action[2] = np.clip(abs(lift_speed), 0.0, max_action)
        return action.astype(np.float32, copy=False), True

    # Case B: safe altitude -> correct XY in the air; hold Z constant.
    xy_correction = kp_xy * xy_error_vec
    xy_correction = np.clip(xy_correction, -max_action, max_action)
    action[0] = xy_correction[0]
    action[1] = xy_correction[1]
    action[2] = 0.0
    return action.astype(np.float32, copy=False), True
