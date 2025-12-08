"""
Generate an offline dataset by rolling out the rule-based Point Maze expert.

Example:
    python gen_offline_data_maze.py \
        --episodes 200 \
        --output offline_dataset_rulebased.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (register PointMaze envs)
import numpy as np

from point_maze_rule_based import (
    MazeLocator,
    direction_to_action,
    extract_success,
    load_maze,
    load_reward_map,
)


BASE_DIR = Path(__file__).resolve().parent

# Defaults mirror point_maze_rule_based.py
DEFAULT_ENV_ID = "PointMaze_Large-v3"
DEFAULT_MAZE_FILE = BASE_DIR / "maze_4room_test.txt"
DEFAULT_REWARD_FILE = BASE_DIR / "maze_4room_reward.txt"
DEFAULT_OUTPUT = BASE_DIR / "offline_dataset_rulebased.pkl"
DEFAULT_EPISODES = 1000
DEFAULT_MAX_STEPS = 1300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out the rule-based expert to build an offline dataset."
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=DEFAULT_ENV_ID,
        help="Gymnasium Point Maze environment id.",
    )
    parser.add_argument(
        "--maze-file",
        type=Path,
        default=DEFAULT_MAZE_FILE,
        help="Path to maze map file.",
    )
    parser.add_argument(
        "--reward-file",
        type=Path,
        default=DEFAULT_REWARD_FILE,
        help="Path to direction/reward map file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Number of episodes to collect.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Step cap per episode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the collected dataset (pickle).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for env reset and action space.",
    )
    parser.add_argument(
        "--min-return",
        type=float,
        default=None,
        help="If set, only keep episodes with total return >= this value.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render while collecting trajectories.",
    )
    return parser.parse_args()


def flatten_goal_observation(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Convert PointMaze's goal-aware dict observation into a flat numpy array.
    Order: observation, achieved_goal, desired_goal, followed by any extras.
    """
    if isinstance(obs, dict):
        ordered_keys: Sequence[str] = ("observation", "achieved_goal", "desired_goal")
        parts: List[np.ndarray] = []

        for key in ordered_keys:
            if key in obs:
                parts.append(np.asarray(obs[key], dtype=np.float32).ravel())

        for key, value in obs.items():
            if key not in ordered_keys:
                parts.append(np.asarray(value, dtype=np.float32).ravel())

        if not parts:
            raise ValueError(
                "Received a dict observation but found no entries to stack."
            )

        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    return np.asarray(obs, dtype=np.float32)


def extract_xy_v(obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[float, float, float, float]:
    """
    Pull (x, y, vx, vy) from a PointMaze observation (dict or flat array).
    """
    if isinstance(obs, dict):
        state = np.asarray(obs.get("observation", []), dtype=np.float32).reshape(-1)
    else:
        state = np.asarray(obs, dtype=np.float32).reshape(-1)

    if state.size < 4:
        raise ValueError("Observation must contain at least x, y, vx, vy.")
    return float(state[0]), float(state[1]), float(state[2]), float(state[3])


def collect_rollouts(
    env: gym.Env,
    locator: MazeLocator,
    reward_map: np.ndarray,
    episodes: int,
    max_steps: int,
    base_seed: int,
    min_return: Optional[float],
    render: bool,
) -> Dict[str, np.ndarray]:
    data: Dict[str, List[np.ndarray]] = {
        "obs": [],
        "act": [],
        "next_observations": [],
        "rew": [],
        "done": [],
        "episode_starts": [],
    }
    all_returns: List[float] = []
    all_lengths: List[int] = []
    kept_returns: List[float] = []
    kept_lengths: List[int] = []
    skipped = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        if isinstance(obs, dict):
            locator.calibrate(obs)
        obs_flat = flatten_goal_observation(obs)

        ep_data: Dict[str, List[np.ndarray]] = {
            "obs": [],
            "act": [],
            "next_observations": [],
            "rew": [],
            "done": [],
            "episode_starts": [],
        }
        ep_return = 0.0
        ep_len = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            x, y, vx, vy = extract_xy_v(obs)
            row, col = locator.world_to_cell(x, y)
            row = int(np.clip(row, 0, reward_map.shape[0] - 1))
            col = int(np.clip(col, 0, reward_map.shape[1] - 1))
            direction = str(reward_map[row, col])
            action = direction_to_action(direction, vx, vy)

            next_obs_raw, env_reward, terminated, truncated, info = env.step(action)
            if render:
                env.render()

            success_flag = extract_success(info)
            if success_flag is None and terminated and not truncated:
                success_flag = True
            if success_flag:
                terminated = True

            reward = float(success_flag) if success_flag is not None else float(env_reward)
            done_flag = terminated or truncated

            next_obs_flat = flatten_goal_observation(next_obs_raw)

            ep_data["obs"].append(obs_flat)
            ep_data["act"].append(action.astype(np.float32))
            ep_data["next_observations"].append(next_obs_flat)
            ep_data["rew"].append(reward)
            ep_data["done"].append(done_flag)
            ep_data["episode_starts"].append(ep_len == 0)

            obs = next_obs_raw
            obs_flat = next_obs_flat
            ep_return += reward
            ep_len += 1

            if max_steps is not None and ep_len >= max_steps:
                break

        all_returns.append(ep_return)
        all_lengths.append(ep_len)

        if min_return is not None and ep_return < min_return:
            skipped += 1
            print(
                f"Episode {ep + 1}/{episodes}: return={ep_return:.2f}, "
                f"length={ep_len} (skipped; below min_return={min_return})"
            )
            continue

        for k in data:
            data[k].extend(ep_data[k])
        kept_returns.append(ep_return)
        kept_lengths.append(ep_len)
        print(
            f"Episode {ep + 1}/{episodes}: return={ep_return:.2f}, length={ep_len} (kept)"
        )

    if kept_lengths:
        print(
            f"Collected {len(data['obs'])} transitions "
            f"({np.mean(kept_lengths):.1f}±{np.std(kept_lengths):.1f} steps/kept-episode)."
        )
        print(
            f"Average kept return: {np.mean(kept_returns):.2f} ± {np.std(kept_returns):.2f}"
        )
    else:
        print("No episodes satisfied min_return; dataset is empty.")

    print(
        f"Summary: kept {len(kept_lengths)} / {episodes} episodes, "
        f"skipped {skipped} (min_return={min_return}). "
        f"All episodes average return: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}"
    )

    return {
        "obs": np.asarray(data["obs"], dtype=np.float32),
        "act": np.asarray(data["act"], dtype=np.float32),
        "next_observations": np.asarray(data["next_observations"], dtype=np.float32),
        "rew": np.asarray(data["rew"], dtype=np.float32),
        "done": np.asarray(data["done"], dtype=bool),
        "episode_starts": np.asarray(data["episode_starts"], dtype=bool),
    }


def main() -> None:
    args = parse_args()

    maze_map = load_maze(args.maze_file)
    reward_map = load_reward_map(args.reward_file)

    env = gym.make(
        args.env_id,
        maze_map=maze_map,
        max_episode_steps=args.max_steps,
        render_mode="human" if args.render else None,
    )
    env.action_space.seed(args.seed)

    locator = MazeLocator(env, maze_map)

    rollouts = collect_rollouts(
        env=env,
        locator=locator,
        reward_map=reward_map,
        episodes=args.episodes,
        max_steps=args.max_steps,
        base_seed=args.seed,
        min_return=args.min_return,
        render=args.render,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(rollouts, f)
    print(f"Saved dataset to {args.output}")

    env.close()


if __name__ == "__main__":
    main()
