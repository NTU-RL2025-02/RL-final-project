from typing import Dict, Sequence, Union

import numpy as np
from gymnasium import Wrapper, ActionWrapper


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


class NoisyActionWrapper(ActionWrapper):
    def __init__(self, env, noise_scale=0.1):
        super().__init__(env)
        self.noise_scale = noise_scale
        self.enabled = True  # 控制要不要加 noise

    def action(self, action):
        if not self.enabled or self.noise_scale == 0:
            return action

        # 連續 action 範例，離散可以改成別的邏輯
        noise = self.noise_scale * np.random.randn(*np.array(action).shape)
        noisy_action = action + noise

        # 夾回 action_space 範圍
        if hasattr(self.env.action_space, "low"):
            noisy_action = np.clip(
                noisy_action, self.env.action_space.low, self.env.action_space.high
            )
        return noisy_action

    def set_noise(self, enabled: bool = True, noise_scale: float | None = None):
        self.enabled = enabled
        if noise_scale is not None:
            self.noise_scale = noise_scale
