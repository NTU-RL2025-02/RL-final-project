#!/usr/bin/env python3
"""Evaluate a trained Point Maze SAC agent and report its success rate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (needed to register the PointMaze envs)
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC


DEFAULT_ENV_ID = "PointMaze_Medium-v3"
DEFAULT_EPISODES = 100
FALLBACK_MAX_STEPS = 1_000

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "best_model_medium.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a saved SAC policy for Point Maze and estimate success rate."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the best_model.zip to evaluate.",
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
        help="Optional step cap per episode (defaults to env.spec.max_episode_steps).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable human render_mode to watch rollouts.",
    )
    return parser.parse_args()


def make_env(env_id: str, render_mode: Optional[str]) -> gym.Env:
    """Create the Point Maze environment with flattened observations."""
    return FlattenObservation(gym.make(env_id, render_mode=render_mode))


def extract_success(info: dict) -> Optional[bool]:
    """
    Extract a success flag from the info dict.
    Handles common keys and array-like values; returns None if no flag is present.
    """
    for key in ("success", "is_success", "goal_achieved", "goal_met", "goal_reached"):
        if key in info:
            try:
                value = np.array(info[key]).astype(bool)
                return bool(value.any())
            except Exception:
                return bool(info[key])
    return None


def rollout_episode(
    model: SAC, env: gym.Env, step_limit: int
) -> Tuple[bool, bool]:
    """Run a single episode; returns (episode_success, success_key_observed)."""
    obs, info = env.reset()
    saw_success_key = False
    episode_success = False
    terminated = False
    truncated = False

    for _ in range(step_limit):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        success_flag = extract_success(info)
        saw_success_key = saw_success_key or success_flag is not None
        if success_flag:
            episode_success = True
            break
        if terminated or truncated:
            break

    # If the env never reports success, treat a natural termination as success.
    if not saw_success_key and terminated and not truncated:
        episode_success = True

    return episode_success, saw_success_key


def run_rollouts(
    model_path: Path,
    env_id: str,
    episodes: int,
    max_steps: Optional[int],
    render: bool,
) -> Tuple[int, int]:
    env = make_env(env_id, render_mode="human" if render else None)
    step_limit = (
        max_steps
        or (env.spec.max_episode_steps if env.spec and env.spec.max_episode_steps else None)
        or FALLBACK_MAX_STEPS
    )

    model = SAC.load(model_path)

    successes = 0
    fallback_successes = 0
    for _ in range(episodes):
        episode_success, saw_success_key = rollout_episode(model, env, step_limit)
        successes += int(episode_success)
        if episode_success and not saw_success_key:
            fallback_successes += 1
            

    env.close()
    return successes, fallback_successes


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    successes, fallback_successes = run_rollouts(
        model_path=model_path,
        env_id=args.env_id,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
    )

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
