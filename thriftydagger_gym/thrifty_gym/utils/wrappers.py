from typing import Dict, Sequence, Union

import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box
from gymnasium.spaces import flatten_space


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
        


    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # FIXME: I am not sure whether to put "and done" here
        self.success = reward > 0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.success = False
        obs, info = super().reset(**kwargs)
        return obs, info

    def is_success(self):
        return self.success
