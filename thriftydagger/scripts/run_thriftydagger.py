# Script for running ThriftyDAgger
from thrifty.algos.thriftydagger import thrifty, generate_offline_data
from thrifty.algos.lazydagger import lazy
from thrifty.utils.run_utils import setup_logger_kwargs
import gymnasium
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

# FIXME: commented out to disable MJGUI-related behavior
# from robosuite.devices.mjgui import MJGUI
from thrifty.utils.hardcoded_nut_assembly import HardcodedPolicy
from robosuite.wrappers import VisualizationWrapper
from robosuite.wrappers import GymWrapper

# from robosuite.devices import Keyboard
import numpy as np
import sys
import time
import wandb

from thrifty.robomimic_expert import RobomimicExpert
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils
from robomimic.utils.file_utils import env_from_checkpoint, policy_from_checkpoint

# 這裡用你搬到比較短路徑的 expert model
# 路徑是相對於你執行 python 的地方（目前你是在 thriftydagger/scripts 底下跑）


env_robomimic, ckpt_dict = env_from_checkpoint(ckpt_path="models/model_epoch_2000_low_dim_v15_success_0.5.pth", render=False)
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


class ObsCachingWrapper:
    """
    Lightweight wrapper that caches raw robosuite dict observations.
    Not a Gymnasium wrapper because the base robosuite env is not a Gym Env.
    """

    def __init__(self, env):
        self.env = env
        self.latest_obs_dict = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.latest_obs_dict = obs
        return obs

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        self.latest_obs_dict = obs
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class CustomWrapper(gymnasium.Env):
    def __init__(self, env, render):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gripper_closed = False
        self.viewer = env.viewer
        self.robots = env.robots
        self._render = render

    def _step(self, action):
        """
        Normalize step outputs to (obs, reward, done, info) even if the base env
        follows the Gymnasium API and returns terminated/truncated separately.
        """
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        return result

    def reset(self):
        res = self.env.reset()  # o ?O obs ?V?q (23,)
        o = res[0] if isinstance(res, tuple) else res
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = -1
        for _ in range(10):
            o, r, d, info = self._step(settle_action)
            # print(o, r, d, info)  # ?u???n debug ?A?L
            self.render()
        self.gripper_closed = False
        return o

    def step(self, action):
        # abstract 10 actions as 1 action
        # get rid of x/y rotation, which is unintuitive for remote teleop
        action_ = action.copy()
        action_[3] = 0.0
        action_[4] = 0.0
        self._step(action_)
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = action[-1]
        for _ in range(10):
            r1, r2, r3, r4 = self._step(settle_action)
            self.render()
        if action[-1] > 0:
            self.gripper_closed = True
        else:
            self.gripper_closed = False
        return r1, r2, r3, r4

    def _check_success(self):
        return self.env._check_success()

    def render(self):
        if self._render:
            self.env.render()


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
    # 你可以下 --environment Square，就會自動用 NutAssemblySquare + Panda
    parser.add_argument("--environment", type=str, default="NutAssembly")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--hgdagger", action="store_true")
    parser.add_argument("--lazydagger", action="store_true")
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
    render = not args.no_render

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # ---- 決定 robosuite env 名字 & 機器人型號 ----
    if args.environment == "Square":
        # 我們的 Square 任務，其實是 robosuite 的 NutAssemblySquare，用 Panda
        robosuite_env_name = "NutAssemblySquare"
        robots = "Panda"
    else:
        # 其他情況就直接用 args.environment
        robosuite_env_name = args.environment
        robots = "UR5e"

    # 共用的 config
    config = {
        "env_name": robosuite_env_name,
        "robots": robots,
        "controller_configs": {
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
        },
    }
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
            "algo": (
                "hgdagger"
                if args.hgdagger
                else (
                    "lazydagger"
                    if args.lazydagger
                    else "thrifty_q" if True else "thrifty"
                )
            ),
            "algo_sup": args.algo_sup,
            "gen_data": args.gen_data,
            "controller_configs": config["controller_configs"],
        },
    )

    # 建立 robosuite 環境
    # env = suite.make(
    #     **config,
    #     has_renderer=render,
    #     has_offscreen_renderer=False,
    #     render_camera="agentview",
    #     ignore_done=True,
    #     use_camera_obs=False,  # low_dim expert，不用影像
    #     reward_shaping=True,
    #     control_freq=20,
    #     hard_reset=True,
    #     use_object_obs=True,
    # )

    obs_cacher = ObsCachingWrapper(env_robomimic.env)
    if isinstance(expert_pol, RobomimicExpert):
        print("Binding environment wrapper to RobomimicExpert...")
        expert_pol.set_env(obs_cacher)
    env = GymWrapper(
        obs_cacher,
        # keys=[
        #     "robot0_eef_pos",
        #     "robot0_eef_quat",
        #     "robot0_gripper_qpos",
        #     "object",
        # ],
    )
    env = VisualizationWrapper(env, indicator_configs=None)
    env = CustomWrapper(env, render=render)
    # o1 = env.env.env.step([0, 0, 0, 0, 0, 0, 0])[0]
    # o2 = env.env.env.env.latest_obs_dict
    # o3 = env_robomimic
    # print("\033[32m", o2, "\033[0m")
    # print("\033[32m", o1, "\033[0m")

    # from sys import exit; exit();

    arm_ = "right"
    config_ = "single-arm-opposed"
    # input_device = Keyboard(env, pos_sensitivity=0.5, rot_sensitivity=3.0)
    if render:
        pass
        # env.viewer.add_keypress_callback(input_device.on_press)
        # env.viewer.add_keyup_callback(input_device.on_release)
        # env.viewer.add_keyrepeat_callback(input_device.on_press)
    active_robot = env.robots[arm_ == "left"]

    def hg_dagger_wait():
        # for HG-dagger, repurpose the 'Z' key (action elem 3) for starting/ending interaction
        # FIXME: commented out to disable MJGUI-related behavior
        # for _ in range(10):
        #     a, _ = MJGUI.input2action(
        #         device=input_device,
        #         robot=active_robot,
        #         active_arm=arm_,
        #         env_configuration=config_,
        #     )
        #     env.render()
        #     time.sleep(0.001)
        #     if a[3] != 0:  # z is pressed
        #         break
        # return a[3] != 0
        pass

    def human_expert_pol(o):
        a = np.zeros(7)
        if env.gripper_closed:
            a[-1] = 1.0
            # input_device.grasp = (
            #     True  # TODO: find how to alter input_device.grasp to newer robosuite
            # )
        else:
            a[-1] = -1.0
            # input_device.grasp = (
            #     False  # TODO: find how to alter input_device.grasp to newer robosuite
            # )
        a_ref = a.copy()
        # pause simulation if there is no user input (instead of recording a no-op)
        while np.array_equal(a, a_ref):
            # FIXME: commented out to disable MJGUI-related behavior
            # a, _ = MJGUI.input2action(
            #     device=input_device,
            #     robot=active_robot,
            #     active_arm=arm_,
            #     env_configuration=config_,
            # )
            env.render()
            time.sleep(0.001)
        return a

    robosuite_cfg = {"MAX_EP_LEN": 175}  # "INPUT_DEVICE": input_device}

    # 根據參數決定要用哪一種 expert
    if args.algo_sup:
        expert_pol = HardcodedPolicy(env).act
    elif args.hgdagger:
        expert_pol = human_expert_pol
    # 否則就用一開始載進來的 robomimic expert_pol

    if args.gen_data:
        NUM_BC_EPISODES = 30
        generate_offline_data(
            env,
            expert_policy=expert_pol,
            num_episodes=NUM_BC_EPISODES,
            seed=args.seed,
            output_file="robosuite-{}.pkl".format(NUM_BC_EPISODES),
            robosuite=True,
            robosuite_cfg=robosuite_cfg,
        )

    try:
        if args.hgdagger:
            thrifty(
                env,
                iters=args.iters,
                logger_kwargs=logger_kwargs,
                device_idx=args.device,
                target_rate=args.targetrate,
                seed=args.seed,
                expert_policy=expert_pol,
                input_file="robosuite-30.pkl",
                robosuite=True,
                robosuite_cfg=robosuite_cfg,
                num_nets=1,
                hg_dagger=hg_dagger_wait,
                init_model=args.eval,
                max_expert_query=args.max_expert_query,
            )
        elif args.lazydagger:
            lazy(
                env,
                iters=args.iters,
                logger_kwargs=logger_kwargs,
                device_idx=args.device,
                noise=0.0,
                seed=args.seed,
                expert_policy=expert_pol,
                input_file="robosuite-30.pkl",
                robosuite=True,
                robosuite_cfg=robosuite_cfg,
            )
        else:
            thrifty(
                env,
                env_robomimic, 
                iters=args.iters,
                logger_kwargs=logger_kwargs,
                device_idx=args.device,
                target_rate=args.targetrate,
                seed=args.seed,
                expert_policy=expert_pol,
                suboptimal_policy=suboptimal_policy,
                extra_obs_extractor=get_observation,
                input_file="models/model_epoch_2000_low_dim_v15_success_0.5-1000.pkl",
                robosuite=True,
                robosuite_cfg=robosuite_cfg,
                q_learning=True,
                init_model=args.eval,
                max_expert_query=args.max_expert_query,
            )
    finally:
        wandb.finish()
