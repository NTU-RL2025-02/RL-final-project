import json
import numpy as np
import torch

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory


class RobomimicExpert:
    """
    Load a robomimic checkpoint and expose an expert_policy(obs) interface for ThriftyDAgger.
    Expected obs dim: 23 = [ robot0_eef_pos(3), robot0_eef_quat(4), robot0_gripper_qpos(2), object(14) ] 但不一定是這樣！！！
    """

    def __init__(self, ckpt_path, device="cuda"):
        self.device = device

        # 1. Load policy + ckpt_dict
        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=ckpt_path,
            device=device,
            verbose=True,
        )

        # 2. Initialize robomimic obs utils from config
        raw_cfg = self.ckpt_dict["config"]
        if isinstance(raw_cfg, str):
            cfg_dict = json.loads(raw_cfg)
        else:
            cfg_dict = raw_cfg

        algo_name = self.ckpt_dict.get("algo_name", "bc")
        config = config_factory(algo_name, dic=cfg_dict)

        ObsUtils.initialize_obs_utils_with_config(config)
        self.env_wrapper = None

    def set_env(self, env_wrapper):
        """Bind environment wrapper so we can recover raw observations when needed."""
        self.env_wrapper = env_wrapper

    def _flatten_obs_dict(self, obs_dict):
        """Build the expected 23-d observation vector from a raw robosuite obs dict."""

        def pick(keys):
            for k in keys:
                if k in obs_dict:
                    return obs_dict[k]
            return None

        eef_pos = pick(["robot0_eef_pos"])
        eef_quat = pick(["robot0_eef_quat"])
        gripper = pick(["robot0_gripper_qpos"])
        obj = pick(["object", "object-state"])

        missing = [
            name
            for name, val in [
                ("robot0_eef_pos", eef_pos),
                ("robot0_eef_quat", eef_quat),
                ("robot0_gripper_qpos", gripper),
                ("object", obj),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"RobomimicExpert is missing required obs keys: {missing}; available keys: {list(obs_dict.keys())}"
            )

        def ensure_batch(x):
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 1:
                x = x[None, :]
            return x

        eef_pos = ensure_batch(eef_pos)
        eef_quat = ensure_batch(eef_quat)
        gripper = ensure_batch(gripper)
        obj = ensure_batch(obj)

        # Some robosuite versions provide a longer object-state; truncate to the 14 dims used for training.
        if obj.shape[-1] > 14:
            obj = obj[..., :14]

        if not (
            eef_pos.shape[0]
            == eef_quat.shape[0]
            == gripper.shape[0]
            == obj.shape[0]
        ):
            raise ValueError(
                "RobomimicExpert could not align batch dims: "
                f"eef_pos{eef_pos.shape}, eef_quat{eef_quat.shape}, gripper{gripper.shape}, obj{obj.shape}"
            )

        return np.concatenate([eef_pos, eef_quat, gripper, obj], axis=-1)

    def __call__(self, obs):
        """
        obs: observation from Thrifty (numpy array or (obs, info) tuple), shape (23,) or (1,23)
        returns: action vector (7,) for env.step
        """

        # unwrap (obs, info) if needed
        if isinstance(obs, (tuple, list)):
            obs = obs[0]

        obs = np.asarray(obs, dtype=np.float32)

        # ensure batch dimension
        if obs.ndim == 1:
            obs = obs[None, :]

        # Try to use the raw observation dictionary from the caching wrapper
        # This is more reliable than trying to decompose the flattened observation
        if obs.shape[-1] != 23:
            if self.env_wrapper is not None and hasattr(self.env_wrapper, "latest_obs_dict"):
                source_dict = self.env_wrapper.latest_obs_dict
                obs = self._flatten_obs_dict(source_dict)
            else:
                raise ValueError(
                    f"RobomimicExpert expected obs dim 23 but got {obs.shape}; "
                    "could not retrieve raw observation dict from environment wrapper"
                )


        obj = obs[0, 0:14]  
        eef_pos      = obs[0, 14:17]     # (3,)
        eef_quat     = obs[0, 17:21]     # (4,)
        gripper_qpos = obs[0, 21:23]     # (2,)

        # robomimic expects this flat structure (not nested under "obs")
        obs_dict = {
            "robot0_eef_pos": eef_pos,
            "robot0_eef_quat": eef_quat,
            "robot0_gripper_qpos": gripper_qpos,
            "object": obj,
        }

        with torch.no_grad():
            act = self.policy(ob=obs_dict)

        # to numpy
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
        else:
            act = np.asarray(act, dtype=np.float32)

        # drop batch dim if present
        if act.ndim > 1 and act.shape[0] == 1:
            act = act[0]

        return act
