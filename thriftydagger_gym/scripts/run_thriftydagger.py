# run_thriftydagger.py
# Script for running ThriftyDAgger on robosuite environments

# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------

# standard libraries
import numpy as np
import sys
import time
import torch
import wandb

# robosuite
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.wrappers import GymWrapper

# thriftydagger
from thrifty_gym.algos.thriftydagger import thrifty
from thrifty_gym.utils.run_utils import setup_logger_kwargs
from thriftydagger_gym.thrifty_gym.utils.wrappers import (
    LunarLanderSuccessWrapper,
    MazeWrapper,
)

import gymnasium as gym
import gymnasium_robotics
from gymnasium_robotics.envs.maze.maps import U_MAZE

from stable_baselines3 import SAC


class SB3Expert:
    """Wrap a Stable-Baselines3 policy to match the expected expert API."""

    def __init__(self, model):
        self.model = model

    def start_episode(self):
        return

    def __call__(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def main(args):
    # ---- load expert policy ----
    # 這裡用你搬到比較短路徑的 expert model
    # 路徑是相對於你執行 python 的地方（目前你是在 thriftydagger/scripts 底下跑）
    device = "cuda" if torch.cuda.is_available() else "cpu"

    expert_model = SAC.load(args.expert_policy_file, device=device)
    expert_pol = SB3Expert(expert_model)
    recovery_model = SAC.load(args.recovery_policy_file, device=device)
    recovery_policy = SB3Expert(recovery_model)

    render = not args.no_render

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    gym.register_envs(gymnasium_robotics)

    # ---- wandb ----
    wandb.init(
        entity="aawrail-RL2025",
        project="final_project_lunar_lander",
        name=args.exp_name,
        config={
            "seed": args.seed,
            "device": args.device,
            "iters": args.iters,
            "target_rate": args.targetrate,
            "environment": args.environment,
            "max_expert_query": args.max_expert_query,
            "expert_policy_file": args.expert_policy_file,
            "recovery_policy_file": args.recovery_policy_file,
            "demonstration_set_file": args.demonstration_set_file,
            "recovery_type": args.recovery_type,
        },
    )

    # ---- 建 env ----
    env = None
    if args.environment == "LunarLander-v3":
        env = gym.make(
            "LunarLander-v3",
            continuous=True,
            gravity=-3.0,
            enable_wind=True,
            wind_power=18.0,
            turbulence_power=1.5,
            render_mode="human" if render else None,
        )
        env = LunarLanderSuccessWrapper(env)  # add success wrapper

    elif args.environment == "PointMaze_UMazeDense-v3":
        env = gym.make(
            "PointMaze_UMazeDense-v3",
            maze_map=U_MAZE,
            continuing_task=False,
            reset_target=False,
            render_mode="human" if render else None,
        )
        env = MazeWrapper(env)  # add success wrapper
    else:
        raise NotImplementedError("This environment is not implemented in this script.")

    max_ep_len = getattr(env, "_max_episode_steps", 1000)
    gym_cfg = {"MAX_EP_LEN": max_ep_len}

    # ---- 主訓練流程 ----
    try:
        thrifty(
            env=env,
            iters=args.iters,
            logger_kwargs=logger_kwargs,
            device_idx=args.device,
            target_rate=args.targetrate,
            seed=args.seed,
            expert_policy=expert_pol,
            recovery_policy=recovery_policy,
            input_file=args.demonstration_set_file,
            robosuite=False,
            gym_cfg=gym_cfg,  # 或者直接傳 None
            q_learning=True,
            init_model=args.eval,
            max_expert_query=args.max_expert_query,
            recovery_type=args.recovery_type,
        )
    except Exception:
        wandb.finish(exit_code=1)
        raise
    else:
        # 正常跑完
        wandb.finish(exit_code=0)
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--iters", type=int, default=20, help="number of DAgger-style iterations"
    )
    parser.add_argument(
        "--targetrate", type=float, default=0.01, help="target context switching rate"
    )

    parser.add_argument(
        "--expert_policy_file",
        type=str,
        default="models/best_model.zip",
        help="filepath to expert model zip file",
    )

    parser.add_argument(
        "--recovery_policy_file",
        type=str,
        default="models/best_model.zip",
        help="filepath to recovery model zip file",
    )

    parser.add_argument(
        "--demonstration_set_file",
        type=str,
        default="models/offline_dataset.pkl",
        help="filepath to expert data pkl file",
    )

    parser.add_argument(
        "--max_expert_query",
        type=int,
        default=2000,
        help="maximum number of expert queries allowed",
    )
    parser.add_argument("--environment", type=str, default="LunarLander-v3")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument(
        "--recovery_type",
        type=str,
        default="five_q",
        choices=["five_q", "q"],
        help="choose recovery policy variant",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="filepath to saved pytorch model to initialize weights",
    )
    args = parser.parse_args()

    main(args)
