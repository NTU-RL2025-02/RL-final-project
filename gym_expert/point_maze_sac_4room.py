"""
Train a Soft Actor-Critic (SAC) expert for Gymnasium's Point Maze environment.
The flow mirrors lunar_lander_sac.py: version prints, vectorized envs, eval and
checkpoint callbacks, best-model video recording, and performance plotting.
"""

import os
import platform
from collections import deque
from importlib.metadata import version

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

import matplotlib.pyplot as plt
import tqdm


class CustomRewardFlattenObservation(FlattenObservation):
    """
    Flatten dict observations and shape reward by BFS distance on the maze grid.
    Each free cell gets its Manhattan distance-to-goal label; reward = -distance.
    """

    def __init__(self, env, maze_map=None):
        super().__init__(env)
        raw_maze_map = (
            maze_map
            if maze_map is not None
            else getattr(self.env.unwrapped, "maze_map", None)
        )
        self.maze_grid = self._normalize_maze_map(raw_maze_map)
        self.distance_field, self.goal_cells = self._compute_distance_field(
            self.maze_grid
        )
        self.grid_rows, self.grid_cols = self.distance_field.shape
        self._cell_size = self._extract_cell_size()
        self._xy_to_rowcol_fn = self._extract_xy_to_rowcol_fn()

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        self._maybe_calibrate_cell_size(obs_dict)
        flat_obs = self.observation(obs_dict)
        return flat_obs, info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        x, y = obs_dict["observation"][:2]

        dist = self._cell_distance(x, y)
        shaped_reward = -dist if np.isfinite(dist) else reward

        flat_obs = self.observation(obs_dict)
        return flat_obs, shaped_reward, terminated, truncated, info

    # ----- distance helpers -----
    def _compute_distance_field(self, maze_grid: np.ndarray):
        """BFS from goal cells to label every reachable cell with grid distance."""
        if maze_grid is None:
            return np.array([[np.inf]], dtype=np.float32), []

        grid = np.array(maze_grid, copy=False)
        rows, cols = grid.shape
        dist = np.full((rows, cols), np.inf, dtype=np.float32)
        queue = deque()
        goals = []

        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                if isinstance(val, str) and val.lower() == "g":
                    dist[r, c] = 0.0
                    queue.append((r, c))
                    goals.append((r, c))

        # If no explicit goal markers, fall back to treating non-walls as goal=0
        if not goals:
            for r in range(rows):
                for c in range(cols):
                    if not self._is_wall(grid[r, c]):
                        dist[r, c] = 0.0
                        queue.append((r, c))
                        goals.append((r, c))
                        break
                if goals:
                    break

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not self._is_wall(grid[nr, nc])
                    and dist[nr, nc] > dist[r, c] + 1
                ):
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
        return dist, goals

    def _cell_distance(self, x: float, y: float) -> float:
        cell = self._world_to_cell(x, y)
        if cell is None:
            return np.inf
        row, col = cell
        if (
            row < 0
            or col < 0
            or row >= self.distance_field.shape[0]
            or col >= self.distance_field.shape[1]
        ):
            return np.inf
        return float(self.distance_field[row, col])

    # ----- coordinate mapping -----
    def _world_to_cell(self, x: float, y: float):
        # Prefer env-provided converters if available.
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

    def _maybe_calibrate_cell_size(self, obs_dict):
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

    def _extract_cell_size(self):
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

    # ----- utilities -----
    @staticmethod
    def _normalize_maze_map(maze_map):
        if maze_map is None:
            return None
        return np.array(maze_map, dtype=object)

    @staticmethod
    def _is_wall(cell) -> bool:
        if isinstance(cell, (int, float)):
            return int(cell) == 1
        if isinstance(cell, str):
            return cell == "1"
        return False

    def _get_goal_position(self, obs_dict):
        """Extract the final goal position from observation/info if available."""
        goal = obs_dict.get("desired_goal")
        if goal is not None:
            goal_arr = np.array(goal, dtype=np.float32).reshape(-1)
            if goal_arr.size >= 2:
                return goal_arr[:2]

        env_target = getattr(self.env.unwrapped, "target", None)
        if env_target is not None:
            target_arr = np.array(env_target, dtype=np.float32).reshape(-1)
            if target_arr.size >= 2:
                return target_arr[:2]
        return None


def safe_version(pkg: str) -> str:
    """Return package version or a placeholder if missing."""
    try:
        return version(pkg)
    except Exception:
        return "not-installed"


print(f"Python Version: {platform.python_version()}")
print(f"Torch Version: {safe_version('torch')}")
print(f"Is Cuda Available: {torch.cuda.is_available()}")
print(f"Cuda Version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"Gymnasium Version: {safe_version('gymnasium')}")
print(f"Numpy Version: {safe_version('numpy')}")
print(f"Stable Baselines3 Version: {safe_version('stable_baselines3')}", flush=True)

RL_TYPE = "SAC"
ENV_ID = "PointMaze_UMaze-v3"  # choose any PointMaze variant you prefer
LOG_DIR = os.path.join("./logs", ENV_ID)
NAME_PREFIX = "point_maze_sac"
MAZE_FILE = "maze_4room.txt"
MAX_EPISODE_STEPS = 1_000  # allow longer rollouts per episode

with open(MAZE_FILE) as file:
    MAZE = [
        list(map(lambda x: int(x) if x in ["0", "1"] else x, line.split()))
        for line in file.readlines()
    ]

# Training/evaluation kwargs keep rendering off for speed; video env enables RGB frames.
TRAIN_EVAL_ENV_KWARGS = {
    "render_mode": None,
    "maze_map": MAZE,
    "max_episode_steps": MAX_EPISODE_STEPS,
}
VIDEO_ENV_KWARGS = {
    "render_mode": "rgb_array",
    "maze_map": MAZE,
    "max_episode_steps": MAX_EPISODE_STEPS,
}

# Vector env settings
N_ENVS = 16
EVAL_ENVS = 1

# Evaluation callback frequency (timesteps) and evaluation episodes
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 10

VIDEO_RECORD_FREQ = 50_000

# Total training steps; Point Maze is sparse so budget generously for expert-quality policy.
TOTAL_TIMESTEPS = 1_200_000

# Visual settings
VIDEO_CAMERA_DISTANCE_SCALE = 3.2  # lift camera higher for clearer recordings

os.makedirs(LOG_DIR, exist_ok=True)


class PeriodicVideoRecorder(BaseCallback):
    """Record a rollout every `record_freq` training timesteps."""

    def __init__(self, record_freq: int, video_length: int = 1_000):
        super().__init__()
        self.record_freq = record_freq
        self.video_length = video_length
        self._last_recorded_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_recorded_step < self.record_freq:
            return True

        name_prefix = f"{NAME_PREFIX}_step{self.num_timesteps}"
        record_video(
            self.model,
            name_prefix=name_prefix,
            video_length=self.video_length,
        )
        self._last_recorded_step = self.num_timesteps
        return True


def main() -> None:
    # Inspect the observation/action spaces after flattening the dict observation.
    sample_env = CustomRewardFlattenObservation(
        gym.make(ENV_ID, **TRAIN_EVAL_ENV_KWARGS),
        maze_map=MAZE,
    )
    print("Observation Space:", sample_env.observation_space)
    print("Action Space:", sample_env.action_space, flush=True)
    sample_env.close()

    # Training and evaluation environments (vectorized).
    env = make_vec_env(
        ENV_ID,
        n_envs=N_ENVS,
        seed=0,
        wrapper_class=CustomRewardFlattenObservation,
        wrapper_kwargs={"maze_map": MAZE},
        env_kwargs=TRAIN_EVAL_ENV_KWARGS,
    )
    env_val = make_vec_env(
        ENV_ID,
        n_envs=EVAL_ENVS,
        seed=1,
        wrapper_class=CustomRewardFlattenObservation,
        wrapper_kwargs={"maze_map": MAZE},
        env_kwargs=TRAIN_EVAL_ENV_KWARGS,
    )

    # Callbacks: save checkpoints and evaluate/track the best model.
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        render=False,
        deterministic=True,
        n_eval_episodes=N_EVAL_EPISODES,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=EVAL_FREQ,
        save_path=os.path.join(LOG_DIR, "checkpoint"),
    )

    video_callback = PeriodicVideoRecorder(record_freq=VIDEO_RECORD_FREQ)

    callback_list = CallbackList([checkpoint_callback, eval_callback, video_callback])

    # Initialize SAC
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
    )

    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=callback_list,
    )

    # Save final model
    model.save(os.path.join(LOG_DIR, "final_model"))

    # Evaluate the latest model
    mean_reward, std_reward = evaluate_policy(
        model, env_val, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
    )
    print(
        f"Final model mean reward: {mean_reward:.2f} +/- {std_reward:.2f}", flush=True
    )

    env.close()
    env_val.close()

    # Load and evaluate the best model saved by EvalCallback
    env_val = make_vec_env(
        ENV_ID,
        n_envs=EVAL_ENVS,
        seed=2,
        wrapper_class=CustomRewardFlattenObservation,
        wrapper_kwargs={"maze_map": MAZE},
        env_kwargs=TRAIN_EVAL_ENV_KWARGS,
    )
    best_model_path = os.path.join(LOG_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        best_model = SAC.load(best_model_path, env=env_val)
        mean_reward, std_reward = evaluate_policy(
            best_model, env_val, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
        )
        print(f"Best model mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        record_video(best_model)
    else:
        print("best_model.zip not found yet; skipping video.")

    env_val.close()
    plot_learning_curve()


def record_video(
    model: SAC,
    name_prefix: str = NAME_PREFIX,
    video_length: int = 1_000,
    env_kwargs=None,
) -> None:
    """Record a short rollout of the model to LOG_DIR."""
    env_kwargs = env_kwargs or VIDEO_ENV_KWARGS
    env = make_vec_env(
        ENV_ID,
        n_envs=1,
        seed=42,
        wrapper_class=CustomRewardFlattenObservation,
        wrapper_kwargs={"maze_map": MAZE},
        env_kwargs=env_kwargs,
    )
    env = VecVideoRecorder(
        env,
        LOG_DIR,
        video_length=video_length,
        record_video_trigger=lambda x: x == 0,
        name_prefix=name_prefix,
    )

    obs = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        env.render()
        if dones.any():
            obs = env.reset()

    env.close()


def plot_learning_curve() -> None:
    """Plot evaluation rewards saved by EvalCallback (evaluations.npz)."""
    eval_path = os.path.join(LOG_DIR, "evaluations.npz")
    if not os.path.exists(eval_path):
        print("evaluations.npz not found; skipping plot.")
        return

    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]

    mean_results = np.mean(results, axis=1)
    std_results = np.std(results, axis=1)

    plt.figure()
    plt.plot(timesteps, mean_results, label="mean reward")
    plt.fill_between(
        timesteps, mean_results - std_results, mean_results + std_results, alpha=0.3
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title(f"{RL_TYPE} on {ENV_ID}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
