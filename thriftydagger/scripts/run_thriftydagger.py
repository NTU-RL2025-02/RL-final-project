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

# robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils
from robomimic.utils.file_utils import env_from_checkpoint, policy_from_checkpoint

# thriftydagger
from thrifty.algos.thriftydagger import thrifty, generate_offline_data
from thrifty.algos.lazydagger import lazy
from thrifty.utils.run_utils import setup_logger_kwargs
from thrifty.utils.hardcoded_nut_assembly import HardcodedPolicy
from thrifty.robomimic_expert import RobomimicExpert
from thrifty.utils.wrapper import ObsCachingWrapper, CustomWrapper


# 這裡用你搬到比較短路徑的 expert model
# 路徑是相對於你執行 python 的地方（目前你是在 thriftydagger/scripts 底下跑）
expert_pol, _ = policy_from_checkpoint(
    device="cuda" if torch.cuda.is_available() else "cpu",
    ckpt_path="models/model_epoch_2000_low_dim_v15_success_0.5.pth",
)
suboptimal_policy, _ = policy_from_checkpoint(
    device="cuda" if torch.cuda.is_available() else "cpu",
    ckpt_path="models/model_epoch_1000.pth",
)
lang_emb = np.load("models/lang_emb.npy")


def get_real_depth_map(env, depth_map):
    """
    Reproduced from https://github.com/ARISE-Initiative/robosuite/blob/c57e282553a4f42378f2635b9a3cbc4afba270fd/robosuite/utils/camera_utils.py#L106
    since older versions of robosuite do not have this conversion from normalized depth values returned by MuJoCo
    to real depth values.
    """
    # Make sure that depth values are normalized
    assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)
    extent = env.sim.model.stat.extent
    far = env.sim.model.vis.map.zfar * extent
    near = env.sim.model.vis.map.znear * extent
    return near / (1.0 - depth_map * (1.0 - near / far))


def get_observation(env, di):
    """
    Get current environment observation dictionary.

    Args:
        di (dict): current raw observation dictionary from robosuite to wrap and provide
            as a dictionary. If not provided, will be queried from robosuite.
    """
    ret = {}
    for k in di:
        if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(
            key=k, obs_modality="rgb"
        ):
            # by default images from mujoco are flipped in height
            ret[k] = di[k][::-1].copy()
        elif (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(
            key=k, obs_modality="depth"
        ):
            # by default depth images from mujoco are flipped in height
            ret[k] = di[k][::-1].copy()
            if len(ret[k].shape) == 2:
                ret[k] = ret[k][..., None]  # (H, W, 1)
            assert len(ret[k].shape) == 3
            # scale entries in depth map to correspond to real distance.
            ret[k] = get_real_depth_map(ret[k])

    # "object" key contains object information
    if "object-state" in di:
        ret["object"] = np.array(di["object-state"])

    for robot in env.robots:
        # add all robot-arm-specific observations. Note the (k not in ret) check
        # ensures that we don't accidentally add robot wrist images a second time
        pf = robot.robot_model.naming_prefix
        for k in di:
            if (
                k.startswith(pf)
                and (k not in ret)
                and (not k.endswith("proprio-state"))
            ):
                ret[k] = np.array(di[k])

    ret[LangUtils.LANG_EMB_OBS_KEY] = np.array(lang_emb)
    return ret


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

    return {
        "env_name": robosuite_env_name,
        "robots": robots,
        "controller_configs": controller_configs,
    }


def create_env(config, render, expert_pol=None):
    """
    建立 robosuite env + wrapper，回傳 (env, active_robot, robosuite_cfg)

    expert_pol: 如果是 RobomimicExpert，就在這裡綁 env。
    """
    env = suite.make(
        **config,
        has_renderer=render,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,  # low_dim expert，不用影像
        reward_shaping=True,
        control_freq=20,
        hard_reset=True,
        use_object_obs=True,
    )

    obs_cacher = ObsCachingWrapper(env)
    if isinstance(expert_pol, RobomimicExpert):
        print("Binding environment wrapper to RobomimicExpert...")
        expert_pol.set_env(obs_cacher)

    env = GymWrapper(
        obs_cacher,
        keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ],
    )
    env = VisualizationWrapper(env, indicator_configs=None)
    env = CustomWrapper(env, render=render)

    arm = "right"
    config_name = "single-arm-opposed"
    active_robot = env.robots[arm == "left"]  # 與你原來邏輯一致

    robosuite_cfg = {"MAX_EP_LEN": 175}
    return env, active_robot, arm, config_name, robosuite_cfg


def main(args):
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
            "algo_sup": args.algo_sup,
            "gen_data": args.gen_data,
            "controller_configs": config["controller_configs"],
        },
    )

    # ---- 建 env ----
    # 假設你的 robomimic expert_pol 是在上面某處初始化好的；如果沒有，就先設成 None
    env, active_robot, arm_, config_, robosuite_cfg = create_env(
        config, render, expert_pol=expert_pol
    )

    # ---- 決定 expert_pol ----
    if args.algo_sup:
        expert = HardcodedPolicy(env).act
    else:
        expert = expert_pol

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
            env,
            iters=args.iters,
            logger_kwargs=logger_kwargs,
            device_idx=args.device,
            target_rate=args.targetrate,
            seed=args.seed,
            expert_policy=expert,
            suboptimal_policy=suboptimal_policy,
            extra_obs_extractor=get_observation,
            input_file="models/model_epoch_2000_low_dim_v15_success_0.5-10.pkl",
            robosuite=True,
            robosuite_cfg=robosuite_cfg,
            q_learning=True,
            init_model=args.eval,
            max_expert_query=args.max_expert_query,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument(
        "--gen_data",
        action="store_true",
        help="True if you want to collect offline human demos",
    )
    parser.add_argument(
        "--iters", type=int, default=5, help="number of DAgger-style iterations"
    )
    parser.add_argument(
        "--targetrate", type=float, default=0.01, help="target context switching rate"
    )
    parser.add_argument(
        "--max_expert_query",
        type=int,
        default=2000,
        help="maximum number of expert queries allowed",
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
