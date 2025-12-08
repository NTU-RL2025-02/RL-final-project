"""
Rule-based Point Maze expert that follows a handcrafted direction map.

Usage:
    python point_maze_rule_based.py --test         # run 20 eval episodes
    python point_maze_rule_based.py --test --render  # render while evaluating
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (register PointMaze envs)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_ID = "PointMaze_Large-v3"
DEFAULT_MAZE_FILE = BASE_DIR.parent / "maze_4room.txt"
DEFAULT_REWARD_FILE = BASE_DIR.parent / "maze_4room_reward.txt"
DEFAULT_EPISODES = 20
DEFAULT_MAX_STEPS = 1300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based Point Maze expert.")
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help="Gymnasium Point Maze environment id.",
    )
    parser.add_argument(
        "--maze-file",
        type=Path,
        default=DEFAULT_MAZE_FILE,
        help="Path to maze map file (default: maze_4room.txt).",
    )
    parser.add_argument(
        "--reward-file",
        type=Path,
        default=DEFAULT_REWARD_FILE,
        help="Path to direction/reward map file (default: maze_4room_reward.txt).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Episodes to roll out when --test is set (default: 20).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable human render mode during rollouts.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run evaluation for the given number of episodes.",
    )
    return parser.parse_args()


def load_maze(path: Path) -> Optional[list]:
    if path is None or not path.exists():
        return None
    with path.open() as f:
        return [
            [int(x) if x in ("0", "1") else x for x in line.split()]
            for line in f.readlines()
        ]


def load_reward_map(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Reward map not found: {path}")
    with path.open() as f:
        grid = [line.split() for line in f.readlines()]
    return np.array(grid, dtype=object)


def flatten_observation(obs: Sequence | dict) -> np.ndarray:
    if isinstance(obs, dict):
        obs_arr = np.array(obs.get("observation", obs.get("achieved_goal", [])))
        return obs_arr.reshape(-1)
    return np.array(obs, dtype=np.float32).reshape(-1)


class MazeLocator:
    def __init__(self, env: gym.Env, maze_map: Optional[list]):
        self.env = env
        self.maze_map = (
            np.array(maze_map, dtype=object) if maze_map is not None else None
        )
        shape = self.maze_map.shape if self.maze_map is not None else (0, 0)
        self.grid_rows, self.grid_cols = shape if shape else (0, 0)
        self._cell_size = self._extract_cell_size()
        self._xy_to_rowcol_fn = self._extract_xy_to_rowcol_fn()
        self.goal_cells = self._find_goal_cells()

    def calibrate(self, obs_dict: dict) -> None:
        if self._cell_size:
            return
        goal_pos = self._get_goal_position(obs_dict)
        if goal_pos is None or not self.goal_cells:
            return
        goal_pos = np.array(goal_pos, dtype=np.float32)
        target_cell = min(
            self.goal_cells,
            key=lambda rc: np.linalg.norm(
                goal_pos - self._cell_center_from_grid(rc[0], rc[1], cell_size=1.0)
            ),
        )
        unscaled_center = self._cell_center_from_grid(*target_cell, cell_size=1.0)
        scales = []
        for axis in range(2):
            denom = unscaled_center[axis]
            if abs(denom) > 1e-6:
                scales.append(abs(goal_pos[axis] / denom))
        if scales:
            self._cell_size = float(np.mean(scales))
        else:
            self._cell_size = 1.0

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        if self._xy_to_rowcol_fn is not None:
            try:
                result = self._xy_to_rowcol_fn(x, y)
            except TypeError:
                result = self._xy_to_rowcol_fn(np.array([x, y]))
            except Exception:
                result = None
            if result is not None:
                try:
                    r, c = result
                    return int(r), int(c)
                except Exception:
                    pass

        cell_size = self._cell_size or 1.0
        col = int(np.floor(x / cell_size + self.grid_cols / 2.0))
        row = int(np.floor(self.grid_rows / 2.0 - y / cell_size))
        col = int(np.clip(col, 0, self.grid_cols - 1))
        row = int(np.clip(row, 0, self.grid_rows - 1))
        return row, col

    def _extract_cell_size(self) -> Optional[float]:
        base_env = self._unwrap_env(self.env)
        for attr in ["maze_size_scaling", "_maze_size_scaling"]:
            val = getattr(base_env, attr, None)
            if val is not None:
                return float(val)
        maze = getattr(base_env, "maze", None)
        if maze is not None:
            for attr in ["maze_size_scaling", "_maze_size_scaling"]:
                val = getattr(maze, attr, None)
                if val is not None:
                    return float(val)
        return None

    def _extract_xy_to_rowcol_fn(self):
        base_env = self._unwrap_env(self.env)
        maze = getattr(base_env, "maze", None)
        candidates = [
            getattr(base_env, "xy_to_rowcol", None),
            getattr(base_env, "_xy_to_rowcol", None),
            getattr(maze, "xy_to_rowcol", None) if maze is not None else None,
            getattr(maze, "_xy_to_rowcol", None) if maze is not None else None,
        ]
        for fn in candidates:
            if callable(fn):
                return fn
        return None

    def _cell_center_from_grid(self, row: int, col: int, cell_size: float = None):
        cs = cell_size or self._cell_size or 1.0
        origin_x = -(self.grid_cols * cs) / 2.0 + cs / 2.0
        origin_y = (self.grid_rows * cs) / 2.0 - cs / 2.0
        x = origin_x + col * cs
        y = origin_y - row * cs
        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def _unwrap_env(env):
        base = env
        while hasattr(base, "env"):
            base = base.env
        return base

    def _find_goal_cells(self):
        if self.maze_map is None:
            return []
        goals = []
        for r in range(self.maze_map.shape[0]):
            for c in range(self.maze_map.shape[1]):
                val = self.maze_map[r, c]
                if isinstance(val, str) and val.lower() == "g":
                    goals.append((r, c))
        return goals

    @staticmethod
    def _get_goal_position(obs_dict: dict):
        goal = obs_dict.get("desired_goal") if isinstance(obs_dict, dict) else None
        if goal is not None:
            goal_arr = np.array(goal, dtype=np.float32).reshape(-1)
            if goal_arr.size >= 2:
                return goal_arr[:2]
        return None


def direction_to_action(direction: str, vx: float, vy: float) -> np.ndarray:
    if direction == "U":
        base = np.array([0.0, 1.0], dtype=np.float32)
        damp = np.array([-vx, 0.0], dtype=np.float32)
    elif direction == "D":
        base = np.array([0.0, -1.0], dtype=np.float32)
        damp = np.array([-vx, 0.0], dtype=np.float32)
    elif direction == "R":
        base = np.array([1.0, 0.0], dtype=np.float32)
        damp = np.array([0.0, -vy], dtype=np.float32)
    elif direction == "L":
        base = np.array([-1.0, 0.0], dtype=np.float32)
        damp = np.array([0.0, -vy], dtype=np.float32)
    else:
        base = np.zeros(2, dtype=np.float32)
        damp = np.array([-vx, -vy], dtype=np.float32)
    return np.clip(base + damp, -1.0, 1.0)


def extract_success(info: dict) -> Optional[bool]:
    for key in ("success", "is_success", "goal_achieved", "goal_met", "goal_reached"):
        if key in info:
            try:
                value = np.array(info[key]).astype(bool)
                return bool(value.any())
            except Exception:
                return bool(info[key])
    return None


def rollout_episode(
    env: gym.Env,
    locator: MazeLocator,
    reward_map: np.ndarray,
    step_limit: int,
    render: bool = False,
) -> bool:
    obs, info = env.reset()
    if isinstance(obs, dict):
        locator.calibrate(obs)

    for _ in range(step_limit):
        obs_vec = flatten_observation(obs)
        if obs_vec.size < 4:
            raise ValueError("Observation must contain at least x, y, vx, vy.")
        x, y, vx, vy = obs_vec[:4]
        row, col = locator.world_to_cell(x, y)
        row = int(np.clip(row, 0, reward_map.shape[0] - 1))
        col = int(np.clip(col, 0, reward_map.shape[1] - 1))
        direction = str(reward_map[row, col])
        action = direction_to_action(direction, vx, vy)

        obs, _, terminated, truncated, info = env.step(action)
        if render:
            env.render()

        success_flag = extract_success(info)
        if success_flag:
            return True
        if terminated or truncated:
            return bool(terminated and not truncated)

    return False


def main() -> None:
    args = parse_args()

    maze_map = load_maze(args.maze_file)
    reward_map = load_reward_map(args.reward_file)

    env_kwargs = {
        "maze_map": maze_map,
        "max_episode_steps": args.max_steps,
        "render_mode": "human" if args.render else None,
    }
    env = gym.make(args.env_id, **env_kwargs)
    locator = MazeLocator(env, maze_map)

    episodes = args.episodes if args.test else 1
    successes = 0
    for _ in range(episodes):
        if rollout_episode(env, locator, reward_map, args.max_steps, args.render):
            successes += 1

    env.close()

    if args.test:
        rate = successes / episodes if episodes else 0.0
        print(f"Success rate: {successes}/{episodes} ({rate:.2%})")


if __name__ == "__main__":
    main()
