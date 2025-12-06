"""
Generate an offline dataset by rolling out a trained SAC policy on LunarLander.

Usage (from thriftydagger_gym/models):
    python gen_offline_data.py \
        --episodes 1000 \
        --output offline_dataset_mazeMedium_1000.pkl

The script defaults to loading `best_model_mediumdense.zip` (trained with SAC)
and the continuous control environment `PointMaze_Medium-v3`.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Roll out SAC expert to build offline dataset."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "./models/best_model_mediumdense.zip",
        help="Path to trained SAC model (.zip).",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="PointMaze_Medium-v3",
        help="Gymnasium environment id used to train the policy.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes to collect.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional cap on steps per episode (defaults to env horizon).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "offline_dataset_mazeMedium_1000.pkl",
        help="Where to store the collected dataset (pickle).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions instead of stochastic ones.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for env reset and action space.",
    )
    parser.add_argument(
        "--min_return",
        type=float,
        default=1,
        help="If set, only keep episodes with total return >= this value.",
    )
    return parser.parse_args()


def flatten_goal_observation(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Convert PointMaze's goal-aware dict observation into a flat numpy array.

    The PointMaze env exposes three keys: ``observation`` (agent state),
    ``achieved_goal`` (current goal state), and ``desired_goal`` (target state).
    We concatenate them in a stable order so downstream learners that expect
    vector inputs (e.g., the BC pretraining code) can consume the data.
    """

    if isinstance(obs, dict):
        ordered_keys: Sequence[str] = ("observation", "achieved_goal", "desired_goal")
        parts: List[np.ndarray] = []

        for key in ordered_keys:
            if key in obs:
                parts.append(np.asarray(obs[key], dtype=np.float32).ravel())

        # Include any extra keys (future-proofing) after the known ordering.
        for key, value in obs.items():
            if key not in ordered_keys:
                parts.append(np.asarray(value, dtype=np.float32).ravel())

        if not parts:
            raise ValueError(
                "Received a dict observation but found no entries to stack."
            )

        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    return np.asarray(obs, dtype=np.float32)


def extract_success(info: Dict[str, Union[bool, np.ndarray]]) -> Optional[bool]:
    """
    Mirror eval_point_maze.py to interpret success flags from env infos.
    Returns True/False when known keys exist, otherwise None.
    """
    candidate_keys: Tuple[str, ...] = (
        "success",
        "is_success",
        "goal_achieved",
        "goal_met",
        "goal_reached",
    )
    for key in candidate_keys:
        if key in info:
            value = info[key]
            try:
                return bool(np.asarray(value).astype(bool).any())
            except Exception:
                return bool(value)
    return None


def collect_rollouts(
    model: SAC,
    env: gym.Env,
    episodes: int,
    max_steps: Optional[int],
    deterministic: bool,
    base_seed: int,
    min_return: Optional[float],
) -> Dict[str, np.ndarray]:
    data: Dict[str, List[np.ndarray]] = {
        "obs": [],
        "act": [],
        "next_observations": [],
        "rewards": [],
        "dones": [],
        "episode_starts": [],
    }
    all_returns: List[float] = []
    all_lengths: List[int] = []
    kept_returns: List[float] = []
    kept_lengths: List[int] = []
    skipped = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        obs = flatten_goal_observation(obs)
        ep_data: Dict[str, List[np.ndarray]] = {
            "obs": [],
            "act": [],
            "next_observations": [],
            "rewards": [],
            "dones": [],
            "episode_starts": [],
        }
        ep_return = 0.0
        ep_len = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs_raw, env_reward, terminated, truncated, info = env.step(action)
            next_obs = flatten_goal_observation(next_obs_raw)

            success_flag = extract_success(info)
            if success_flag is None and terminated and not truncated:
                success_flag = True
            if success_flag:
                terminated = True

            reward = (
                float(success_flag) if success_flag is not None else float(env_reward)
            )
            done_flag = terminated or truncated

            ep_data["obs"].append(obs)
            ep_data["act"].append(action)
            ep_data["next_observations"].append(next_obs)
            ep_data["rewards"].append(reward)
            ep_data["dones"].append(done_flag)
            ep_data["episode_starts"].append(ep_len == 0)

            obs = next_obs
            ep_return += reward
            ep_len += 1

            if max_steps is not None and ep_len >= max_steps:
                # Forcefully end the episode if a custom cap is provided.
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
        "rewards": np.asarray(data["rewards"], dtype=np.float32),
        "dones": np.asarray(data["dones"], dtype=bool),
        "episode_starts": np.asarray(data["episode_starts"], dtype=bool),
    }


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    env = FlattenObservation(gym.make(args.env_id))
    env.action_space.seed(args.seed)

    model: SAC = SAC.load(str(args.model))

    rollouts = collect_rollouts(
        model=model,
        env=env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        base_seed=args.seed,
        min_return=args.min_return,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(rollouts, f)
    print(f"Saved dataset to {args.output}")

    env.close()


if __name__ == "__main__":
    main()
