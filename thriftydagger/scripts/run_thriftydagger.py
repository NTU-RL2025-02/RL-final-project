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

# FIXME: commented out to disable MJGUI-related behavior
# from robosuite.devices.mjgui import MJGUI
# from robosuite.devices import Keyboard

# thriftydagger
from thrifty.algos.thriftydagger import thrifty, generate_offline_data
from thrifty.algos.lazydagger import lazy
from thrifty.utils.run_utils import setup_logger_kwargs
from thrifty.utils.hardcoded_nut_assembly import HardcodedPolicy
from thrifty.utils.wrapper import ObsCachingWrapper, CustomWrapper

def build_robosuite_config(args):
    """
    負責決定 env_name / robots / controller_configs
    """
    # ---- 決定 robosuite env 名字 & 機器人型號 ----
    if args.environment == "Square":
        # 我們的 Square 任務，其實是 robosuite 的 NutAssemblySquare，用 Panda
        robosuite_env_name = "NutAssemblySquare"
        robots = "Panda"
    else:
        # 其他情況就直接用 args.environment
        robosuite_env_name = args.environment
        robots = "UR5e"

    controller_configs = {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_ratio_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "input_type": "delta",
                "input_ref_frame": "base",
                "interpolation": None,
                "ramp_ratio": 0.2,
                "gripper": {"type": "GRIP"},
            }
        },
    }
    print("env:", robosuite_env_name)

    return {
        "env_name": robosuite_env_name,
        "robots": robots,
        "controller_configs": controller_configs,
    }


def create_env(config, render):
    """
    建立 robosuite env + wrapper，回傳 (env, active_robot, robosuite_cfg)
    """
    env = suite.make(
        **config,
        has_renderer=render,
        has_offscreen_renderer=False,
        render_camera="agentview",
        single_object_mode=2,
        nut_type="round",
        ignore_done=True,
        use_camera_obs=False,  # low_dim expert，不用影像
        reward_shaping=True,
        control_freq=20,
        hard_reset=True,
        use_object_obs=True,
    )
    obs_cacher = ObsCachingWrapper(env)
    env = GymWrapper(
        obs_cacher
    )
    env = VisualizationWrapper(env, indicator_configs=None)
    env = CustomWrapper(env, render=render)

    robosuite_cfg = {"MAX_EP_LEN": 300}
    return env, robosuite_cfg


def main(args):
    # ---- load expert policy ----
    # 這裡用你搬到比較短路徑的 expert model
    # 路徑是相對於你執行 python 的地方（目前你是在 thriftydagger/scripts 底下跑）

    render = not args.no_render

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # ---- robosuite config ----
    config = build_robosuite_config(args)

    # ---- wandb ----
    wandb.init(
        entity="aawrail-RL2025",
        project="final_project_exp0",
        name=args.exp_name,
        config={
            "seed": args.seed,
            "device": args.device,
            "iters": args.iters,
            "target_rate": args.targetrate,
            "environment": args.environment,
            "max_expert_query": args.max_expert_query,
            "algo": ("thrifty_q" if True else "thrifty"),
            "expert_policy_file": args.expert_policy_file,
            "recovery_policy_file": args.recovery_policy_file,
            "demonstration_set_file": args.demonstration_set_file,
            "gen_data": args.gen_data,
            "controller_configs": config["controller_configs"],
        },
    )

    # ---- 建 env ----
    env, robosuite_cfg = create_env(
        config, render
    )

    # ---- 決定 expert_pol ----
    if args.algo_sup:
        expert = HardcodedPolicy(env).act

    # ---- 如果要先收 offline data ----
    if args.gen_data:
        NUM_BC_EPISODES = 30
        generate_offline_data(
            env,
            expert_policy=expert,
            num_episodes=NUM_BC_EPISODES,
            seed=args.seed,
            output_file=f"robosuite-{NUM_BC_EPISODES}.pkl",
            robosuite=True,
            robosuite_cfg=robosuite_cfg,
        )

    # ---- 主訓練流程 ----
    try:
        thrifty(
            env=env,
            iters=args.iters,
            logger_kwargs=logger_kwargs,
            device_idx=args.device,
            target_rate=args.targetrate,
            seed=args.seed,
            expert_policy=expert,
            recovery_policy=expert,
            input_file=args.demonstration_set_file,
            robosuite=True,
            robosuite_cfg=robosuite_cfg,
            q_learning=True,
            init_model=args.eval,
            max_expert_query=args.max_expert_query,
            num_test_episodes=args.num_test_episodes
        )
    except Exception:
        wandb.finish(exit_code=1)
        raise
    else:
        # 正常跑完
        wandb.finish(exit_code=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--gen_data",
        action="store_true",
        help="True if you want to collect offline human demos",
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="number of DAgger-style iterations"
    )
    parser.add_argument(
        "--targetrate", type=float, default=0.01, help="target context switching rate"
    )

    parser.add_argument(
        "--expert_policy_file",
        type=str,
        default="models/model_epoch_2000_low_dim_v15_success_0.5.pth",
        help="filepath to expert policy pth file",
    )

    parser.add_argument(
        "--recovery_policy_file",
        type=str,
        default="models/model_epoch_1000.pth",
        help="filepath to recovery policy pth file",
    )

    parser.add_argument(
        "--demonstration_set_file",
        type=str,
        default="models/model_epoch_2000_low_dim_v15_success_0.5-1000.pkl",
        help="filepath to expert data pkl file",
    )

    parser.add_argument(
        "--max_expert_query",
        type=int,
        default=2000,
        help="maximum number of expert queries allowed",
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=10,
        help="number of agent being tested each episode",
    )
    parser.add_argument("--environment", type=str, default="NutAssembly")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="filepath to saved pytorch model to initialize weights",
    )
    parser.add_argument(
        "--algo_sup", action="store_true", help="use an algorithmic supervisor"
    )
    args = parser.parse_args()

    main(args)
