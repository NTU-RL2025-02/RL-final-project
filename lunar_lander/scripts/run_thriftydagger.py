import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

from lunar_lander.thrifty.thriftydagger import thrifty, generate_offline_data
from lunar_lander.thrifty.core import Ensemble
from lunar_lander.thrifty.utils.run_utils import setup_logger_kwargs


class SB3Expert:
    """Wrap a Stable-Baselines3 policy to match the expected expert API."""

    def __init__(self, model):
        self.model = model

    def start_episode(self):
        return

    def __call__(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ThriftyDAgger on LunarLander.")
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--gen_data", action="store_true", help="Collect offline demos only.")
    parser.add_argument("--iters", type=int, default=5, help="Number of DAgger iterations.")
    parser.add_argument("--targetrate", type=float, default=0.01, help="Target switch rate.")
    parser.add_argument("--max_expert_query", type=int, default=2000)
    parser.add_argument("--environment", type=str, default="LunarLanderContinuous-v2")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--eval", type=str, default=None, help="Optional init model path.")
    parser.add_argument("--offline_file", type=Path, default=Path("lunar_lander/offline_dataset.pkl"))
    parser.add_argument("--expert_model", type=Path, default=Path("lunar_lander/best_model.zip"))
    parser.add_argument("--episodes", type=int, default=100, help="Episodes for offline collection.")
    parser.add_argument("--success_threshold", type=float, default=200.0, help="Return threshold for success.")
    return parser.parse_args()


def main():
    args = parse_args()
    render = not args.no_render
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    env = gym.make(args.environment, render_mode="human" if render else None)
    expert_model: SAC = SAC.load(str(args.expert_model))
    expert_pol = SB3Expert(expert_model)
    suboptimal_pol = expert_pol

    if args.gen_data:
        generate_offline_data(
            env,
            expert_policy=expert_pol,
            num_episodes=args.episodes,
            output_file=str(args.offline_file),
            seed=args.seed,
        )
        return

    thrifty(
        env,
        iters=args.iters,
        actor_critic=Ensemble,
        seed=args.seed,
        logger_kwargs=logger_kwargs,
        device_idx=args.device,
        target_rate=args.targetrate,
        expert_policy=expert_pol,
        suboptimal_policy=suboptimal_pol,
        input_file=str(args.offline_file),
        success_threshold=args.success_threshold,
        max_expert_query=args.max_expert_query,
        init_model=args.eval,
    )


if __name__ == "__main__":
    main()
