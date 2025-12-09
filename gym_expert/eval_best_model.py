"""
Run rollouts of a trained Point Maze SAC policy and report success rate.
Defaults match the 4-room setup used in point_maze_aaw.py.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (register PointMaze envs)
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC
from point_maze_aaw import CustomRewardFlattenObservation
from thrifty_gym.utils.wrappers import MazeWrapper
from thrifty_gym.maze import FOUR_ROOMS_21_21_RANDOM_START


DEFAULT_MODEL_PATH = Path("logs/PointMaze_Large-v3/best_model.zip")
DEFAULT_ENV_ID = "PointMaze_Large-v3"
DEFAULT_EPISODES = 20
DEFAULT_MAX_STEPS = 1300


def rollout_episode(model: SAC, env: gym.Env, step_limit: int) -> Tuple[bool, bool]:
    obs, _ = env.reset()
    saw_success_key = False
    episode_success = False
    step_count = 0

    for _ in range(step_limit):
        step_count += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        success_flag = env.is_success()
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


def main(args) -> None:
    model_path = args.model.expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    maze_map = FOUR_ROOMS_21_21_RANDOM_START

    env = gym.make(
        id=args.env_id,
        maze_map=maze_map,
        render_mode="human" if args.render else None,
        max_episode_steps=args.max_steps,
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

    env = FlattenObservation(env)
    env = MazeWrapper(env, maze=maze_map, touch_wall_distance=0.15)

    model = SAC.load(model_path)

    successes = 0
    fallback_successes = 0
    for i in range(args.episodes):
        env.seed(i)
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
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SAC policy for Point Maze and report success rate."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to model zip file (default: logs/PointMaze_Large-v3/best_model.zip)",
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
    args = parser.parse_args()

    main(args)
