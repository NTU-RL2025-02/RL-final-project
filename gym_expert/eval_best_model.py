"""
Run rollouts of a trained Point Maze SAC policy and report success rate.
Defaults match the 4-room setup used in point_maze_aaw.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (register PointMaze envs)
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC


DEFAULT_MODEL_PATH = Path("logs/PointMaze_UMaze-v3/best_model.zip")
DEFAULT_ENV_ID = "PointMaze_UMaze-v3"
DEFAULT_EPISODES = 20
DEFAULT_MAZE_FILE = Path("maze_4room.txt")
DEFAULT_MAX_STEPS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SAC policy for Point Maze and report success rate."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to best_model.zip (default: logs/PointMaze_UMaze-v3/best_model.zip)",
    )
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help="Gymnasium Point Maze environment id to use.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Number of rollouts to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional step cap per episode (defaults to env.spec.max_episode_steps or 1000).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable human render_mode to watch rollouts.",
    )
    parser.add_argument(
        "--maze-file",
        type=Path,
        default=DEFAULT_MAZE_FILE,
        help="Path to maze map file; if missing, uses env default maze.",
    )
    return parser.parse_args()


def load_maze(maze_file: Path) -> Optional[list]:
    if maze_file is None or not maze_file.exists():
        return None
    with maze_file.open() as f:
        return [
            [int(x) if x in ("0", "1") else x for x in line.split()]
            for line in f.readlines()
        ]


def make_env(env_id: str, maze_map: Optional[list], render_mode: Optional[str]) -> gym.Env:
    env_kwargs = {"render_mode": render_mode}
    if maze_map is not None:
        env_kwargs["maze_map"] = maze_map
        env_kwargs["max_episode_steps"] = DEFAULT_MAX_STEPS
    return FlattenObservation(gym.make(env_id, **env_kwargs))


def extract_success(info: dict) -> Optional[bool]:
    for key in ("success", "is_success", "goal_achieved", "goal_met", "goal_reached"):
        if key in info:
            try:
                value = np.array(info[key]).astype(bool)
                return bool(value.any())
            except Exception:
                return bool(info[key])
    return None


def rollout_episode(model: SAC, env: gym.Env, step_limit: int) -> Tuple[bool, bool]:
    obs, info = env.reset()
    saw_success_key = False
    episode_success = False
    step_count = 0

    for _ in range(step_limit):
        step_count += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        success_flag = extract_success(info)
        saw_success_key = saw_success_key or success_flag is not None
        if success_flag:
            print(f"Success detected via info flag. step = {step_count}")
            episode_success = True
            break
        if terminated or truncated:
            print(f"Episode ended. steps = {step_count}")
            break

    if not saw_success_key and terminated and not truncated:
        episode_success = True

    return episode_success, saw_success_key


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    maze_map = load_maze(args.maze_file)
    env = make_env(
        env_id=args.env_id,
        maze_map=maze_map,
        render_mode="human" if args.render else None,
    )

    step_limit = (
        args.max_steps
        or (
            env.spec.max_episode_steps
            if env.spec and env.spec.max_episode_steps
            else None
        )
        or DEFAULT_MAX_STEPS
    )

    model = SAC.load(model_path)

    successes = 0
    fallback_successes = 0
    for _ in range(args.episodes):
        episode_success, saw_success_key = rollout_episode(model, env, step_limit)
        successes += int(episode_success)
        if episode_success and not saw_success_key:
            fallback_successes += 1

    env.close()

    success_rate = successes / args.episodes if args.episodes else 0.0
    print(f"Evaluated {args.episodes} episodes on {args.env_id}")
    print(f"Successes: {successes}/{args.episodes} ({success_rate:.2%})")
    if fallback_successes:
        print(
            f"{fallback_successes} successes counted via termination fallback "
            "because no success flag was provided in info."
        )


if __name__ == "__main__":
    main()
