"""
Data-driven recovery policy that looks up a nearest-neighbor state in the
expert replay buffer (flattened observations) and steers the gripper toward
the corresponding relative pose w.r.t. the square nut.
"""

import numpy as np

# Helpers to extract the pieces we need from raw or flattened observations.
def get_recovery_state(obs):
    """
    Extract the end-effector pose and square nut pose from a robosuite obs dict.

    Returns:
        dict with keys:
            - eef_pos: np.ndarray shape (3,) float32
            - square_pos: np.ndarray shape (3,) float32
            - square_quat: np.ndarray shape (4,) float32
    Raises:
        ValueError if any required field is missing or has the wrong length.
    """
    required = {
        "robot0_eef_pos": 3,
        "SquareNut_pos": 3,
        "SquareNut_quat": 4,
    }
    missing = [k for k in required if k not in obs]
    if missing:
        raise ValueError(f"Missing keys in obs: {missing}")

    def grab(key, expected_len):
        arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        if arr.shape[0] != expected_len:
            raise ValueError(f"{key} expected length {expected_len}, got shape {arr.shape}")
        return arr

    return {
        "eef_pos": grab("robot0_eef_pos", required["robot0_eef_pos"]),
        "square_pos": grab("SquareNut_pos", required["SquareNut_pos"]),
        "square_quat": grab("SquareNut_quat", required["SquareNut_quat"]),
    }


# Index map for the flattened observation used by GymWrapper in run_thriftydagger.py.
_EEF_POS_SLICE = slice(0, 3)
_OBJECT_START = 3 + 4 + 2  # eef_pos(3) + eef_quat(4) + gripper_qpos(2) = 9
_SQUARE_POS_SLICE = slice(_OBJECT_START + 7, _OBJECT_START + 10)  # object offsets 7:10
_SQUARE_QUAT_SLICE = slice(_OBJECT_START + 10, _OBJECT_START + 14)  # object offsets 10:14


def get_recovery_state_from_flat(flat_obs):
    """
    Extract eef_pos and SquareNut pose from the flattened observation vector stored
    in the replay buffer (GymWrapper with keys [eef_pos, eef_quat, gripper_qpos, object]).

    Assumes object-state ordering observed in robosuite NutAssemblySquare:
        SquareNut_to_eef_pos(3), SquareNut_to_eef_quat(4), SquareNut_pos(3),
        SquareNut_quat(4), RoundNut_to_eef_pos(3), RoundNut_to_eef_quat(4),
        RoundNut_pos(3), RoundNut_quat(4)
    """
    flat_obs = np.asarray(flat_obs, dtype=np.float32).reshape(-1)
    min_len = _OBJECT_START + 14  # need through SquareNut_quat
    if flat_obs.shape[0] < min_len:
        raise ValueError(
            f"Flat obs too short (len={flat_obs.shape[0]}), expected at least {min_len} "
            "to extract eef and SquareNut pose."
        )

    eef_pos = flat_obs[_EEF_POS_SLICE]
    square_pos = flat_obs[_SQUARE_POS_SLICE]
    square_quat = flat_obs[_SQUARE_QUAT_SLICE]

    return {
        "eef_pos": eef_pos.astype(np.float32, copy=False),
        "square_pos": square_pos.astype(np.float32, copy=False),
        "square_quat": square_quat.astype(np.float32, copy=False),
    }


class DataDrivenRecovery:
    """
    Nearest-neighbor recovery:
      1) Compute current relative vector (eef - nut) from the flattened obs.
      2) Find the expert state in replay_buffer with closest relative vector.
      3) Map that relative offset to the current nut position to form a target.
      4) Output a 7D action pointing toward the target (orientation/gripper zeroed).
    """

    def __init__(
        self,
        replay_buffer,
        action_gain: float = 5.0,
        max_action: float = 1.0,
        min_z_margin: float = 0.0,
    ):
        """
        Args:
            replay_buffer: buffer with attributes obs_buf (numpy) and size.
            action_gain: P-gain for position error to action scaling.
            max_action: clamp each action dimension to [-max_action, max_action].
            min_z_margin: require expert eef z >= current_z + min_z_margin when filtering.
        """
        self.replay_buffer = replay_buffer
        self.action_gain = action_gain
        self.max_action = max_action
        self.min_z_margin = min_z_margin

    def _parse_flat(self, flat_obs):
        """Helper to extract eef/nut pose from a flattened obs vector."""
        state = get_recovery_state_from_flat(flat_obs)
        return state["eef_pos"], state["square_pos"], state["square_quat"]

    def _build_action(self, current_eef, target_pos):
        """P-control toward target position; orientation+gripper untouched (zeros)."""
        delta = target_pos - current_eef
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(self.action_gain * delta, -self.max_action, self.max_action)
        return action

    def __call__(self, current_obs):
        """
        Args:
            current_obs: flattened observation vector (numpy-like) from GymWrapper.
        Returns:
            action (np.ndarray shape (7,)) or None if no valid neighbor is found.
        """
        flat_curr = np.asarray(current_obs, dtype=np.float32).reshape(-1)
        curr_eef, curr_nut, _ = self._parse_flat(flat_curr)
        curr_rel = curr_eef - curr_nut

        buf = self.replay_buffer
        if not hasattr(buf, "obs_buf") or buf.size == 0:
            return None

        expert_obs = buf.obs_buf[: buf.size]
        # vectorized extraction of eef and nut from flattened buffer
        eef_expert = expert_obs[:, 0:3]
        # nut is at indices 16:19 in our GymWrapper flattening (object offsets 7:10)
        nut_expert = expert_obs[:, 16:19]

        rel_expert = eef_expert - nut_expert

        # Filter: expert z should be above current eef z by margin
        safe_mask = eef_expert[:, 2] >= (curr_eef[2] + self.min_z_margin)
        if not np.any(safe_mask):
            return None

        rel_safe = rel_expert[safe_mask]
        nut_safe = nut_expert[safe_mask]
        eef_safe = eef_expert[safe_mask]

        # Nearest neighbor in relative space
        diffs = rel_safe - curr_rel
        dists = np.linalg.norm(diffs, axis=1)
        idx = np.argmin(dists)

        # Map expert relative offset to current nut position
        best_rel = rel_safe[idx]
        target_pos = curr_nut + best_rel

        return self._build_action(curr_eef, target_pos)
