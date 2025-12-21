from typing import Dict, Sequence, Union, List
import math
import numpy as np
from gymnasium import Wrapper, ActionWrapper


def nearest_wall_distance(x: float, y: float, wall_indices, n_cols, n_rows) -> float:
    """
    walls[i, j] 為 True/1 表示該格是牆，大小為 n x n。
    世界座標範圍為 x, y ∈ [-n/2, n/2]。
    回傳點 (x, y) 到最近一格牆的歐式距離。
    """

    if wall_indices.size == 0:
        return math.inf  # 沒有牆

    min_dist = math.inf

    for i, j in wall_indices:
        # 對應到世界座標的 rectangle
        x_min = j - n_cols / 2
        y_max = -i + n_rows / 2
        x_max = x_min + 1.0
        y_min = y_max - 1.0

        # 點到 rectangle 的距離
        if x < x_min:
            dx = x_min - x
        elif x > x_max:
            dx = x - x_max
        else:
            dx = 0.0

        if y < y_min:
            dy = y_min - y
        elif y > y_max:
            dy = y - y_max
        else:
            dy = 0.0

        dist = math.hypot(dx, dy)
        if dist < min_dist:
            min_dist = dist

    return min_dist


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
    def __init__(self, env, maze=None, touch_wall_distance: float = 0.15):
        super().__init__(env)
        self.success = False
        if maze:
            self.maze = np.asarray(maze)
            self.n_rows, self.n_cols = self.maze.shape

            wall_indices = []
            # 找出所有牆的 index
            for i, row in enumerate(self.maze):
                for j, entry in enumerate(row):
                    if entry == "1" or entry == 1:
                        wall_indices.append([i, j])
            self.wall_indices = np.array(wall_indices)

        self.touch_wall_distance = touch_wall_distance

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        x = obs[4]
        y = obs[5]
        vx = obs[6]
        vy = obs[7]

        if self.wall_indices is not None:
            dist = nearest_wall_distance(
                x, y, self.wall_indices, self.n_cols, self.n_rows
            )
            if dist < self.touch_wall_distance:
                info["touched_wall"] = True
                terminated = True
                reward = -1.0

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
    def __init__(
        self,
        env,
        noise_scale=0.1,
    ):
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
