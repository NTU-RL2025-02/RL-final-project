from typing import Dict, Sequence, Union

import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box
from gymnasium.spaces import flatten_space


def _flatten_maze_observation(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Flatten PointMaze's dict observation into a single float32 vector.

    Keeps a fixed ordering (observation -> achieved_goal -> desired_goal) so the
    online env matches the offline dataset created via gen_offline_data_maze.py.
    """
    if isinstance(obs, dict):
        ordered_keys: Sequence[str] = ("observation", "achieved_goal", "desired_goal")
        parts = []

        for key in ordered_keys:
            if key in obs:
                parts.append(np.asarray(obs[key], dtype=np.float32).ravel())

        for key, value in obs.items():
            if key not in ordered_keys:
                parts.append(np.asarray(value, dtype=np.float32).ravel())

        if not parts:
            raise ValueError(
                "Dict observation did not contain any flattenable entries."
            )

        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    return np.asarray(obs, dtype=np.float32)


class LunarLanderSuccessWrapper(Wrapper):
    """
    Wrapper to track success in LunarLander environment.
    Success is defined as achieving an episode reward of at least 200.
    """

    def __init__(self, env):
        super().__init__(env)
        self.success = False
        self.ep_reward = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        self.ep_reward += reward

        # FIXME: I am not sure whether to put "and done" here
        self.success = (self.ep_reward >= 200.0) and done

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.success = False
        self.ep_reward = 0.0
        return super().reset(**kwargs)

    def is_success(self):
        return self.success


class MazeWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.success = False
        flattened_space = flatten_space(env.observation_space)
        low = np.full(flattened_space.shape, -np.inf, dtype=np.float32)
        high = np.full(flattened_space.shape, np.inf, dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = _flatten_maze_observation(obs)
        # FIXME: I am not sure whether to put "and done" here
        self.success = reward > 0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.success = False
        obs, info = super().reset(**kwargs)
        return _flatten_maze_observation(obs), info

    def is_success(self):
        return self.success
