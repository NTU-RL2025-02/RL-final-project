from typing import Dict, Sequence, Union
import math
import numpy as np
from gymnasium import Wrapper, ActionWrapper


def nearest_wall_distance(walls: np.ndarray, x: float, y: float, env) -> float:
    """
    walls[i, j] 為 True/1 表示該格是牆，大小為 n x n。
    世界座標範圍為 x, y ∈ [-n/2, n/2]。
    回傳點 (x, y) 到最近一格牆的歐式距離。
    """
    n_rows, n_cols = walls.shape
    assert n_rows == n_cols, "這裡假設是正方形迷宮"
    n = n_rows

    # 找出所有牆的 index
    wall_indices = np.argwhere(walls)
    if wall_indices.size == 0:
        return math.inf  # 沒有牆

    min_dist = math.inf

    for i, j in wall_indices:
        # 對應到世界座標的 rectangle
        x_min, y_max = env.cell_rowcol_to_xy(i, j)
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
    def __init__(self, env, maze=None, touch_wall_distance: float = 0.1):
        super().__init__(env)
        self.success = False
        if maze:
            self.maze = np.asarray(maze)
        self.touch_wall_distance = touch_wall_distance

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        x = obs[0]
        y = obs[1]
        vx = obs[2]
        vy = obs[3]

        if self.maze:
            dist = nearest_wall_distance(self.maze, x, y, self.env)
            if dist < 0.1:
                info["touched_wall"] = True
                terminated = True

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

    def cell_rowcol_to_xy(i, j):
        return super().cell_rowcol_to_xy(i, j)
