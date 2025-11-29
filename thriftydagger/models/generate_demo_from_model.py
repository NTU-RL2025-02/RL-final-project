import gymnasium as gym
import pickle, numpy as np, torch
import robosuite as suite
from robosuite.wrappers import GymWrapper, VisualizationWrapper
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils
from robomimic.algo import algo_factory
from robomimic.utils.file_utils import env_from_checkpoint, policy_from_checkpoint
from thrifty.robomimic_expert import RobomimicExpert
from copy import deepcopy

# copy from robosuite.GymWrapper
def _flatten_obs(obs_dict, verbose=False):
    """
    Filters keys of interest out and concatenate the information.

    Args:
        obs_dict (OrderedDict): ordered dictionary of observations
        verbose (bool): Whether to print out to console as observation keys are processed

    Returns:
        np.array: observations flattened into a 1d array
    """
    return np.concat([obs_dict['robot0_proprio-state'], obs_dict['object-state']])

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
class CustomWrapper(gym.Env):
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


lang_emb = np.load("models/lang_emb.npy")
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

# def raw_observation_to_thrifty_dagger_observation(obs_dict):
#     return {
#         "robot0_eef_pos": obs_dict["robot0_eef_pos"],
#         "robot0_eef_quat": obs_dict["robot0_eef_quat"],
#         "robot0_gripper_qpos": obs_dict["robot0_gripper_qpos"],
#         "object": obs_dict["object-state"],
#     }


model_name = "model_epoch_2000_low_dim_v15_success_0.5"
ckpt = f"models/{model_name}.pth"
robomimic_env, ckpt_dict = env_from_checkpoint(ckpt_path=ckpt, render=False)
policy, _ = policy_from_checkpoint(device="cuda" if torch.cuda.is_available() else "cpu", ckpt_dict=ckpt_dict)

config = {
    "env_name": "NutAssemblySquare",
    "robots": "Panda",
    "camera_depths": False,
    "camera_heights": 84,
    "camera_widths": 84,
    "controller_configs": {
        "body_parts": {
            "right": {
                "control_delta": True,
                "damping": 1,
                "damping_limits": [
                    0,
                    10
                ],
                "gripper": {
                    "type": "GRIP"
                },
                "impedance_mode": "fixed",
                "input_max": 1,
                "input_min": -1,
                "interpolation": None, 
                "input_ref_frame": "world",
                "kp": 150,
                "kp_limits": [
                    0,
                    300
                ],
                "output_max": [
                    0.05,
                    0.05,
                    0.05,
                    0.5,
                    0.5,
                    0.5
                ],
                "output_min": [
                    -0.05,
                    -0.05,
                    -0.05,
                    -0.5,
                    -0.5,
                    -0.5
                ],
                "ramp_ratio": 0.2,
                "type": "OSC_POSE",    
            }
        },
        "type": "BASIC"
    },
}
# 建立 robosuite 環境
# env = suite.make(
#     **config,
#     has_renderer=False,
#     has_offscreen_renderer=False,
#     render_camera="agentview",
#     ignore_done=True,
#     use_camera_obs=False,  # low_dim expert，不用影像
#     reward_shaping=False,
#     control_freq=20,
#     lite_physics=False,
#     hard_reset=True,
#     use_object_obs=True,
# )
obs_cacher = ObsCachingWrapper(robomimic_env.env)
env = GymWrapper(obs_cacher)
env = VisualizationWrapper(env, indicator_configs=None)
env = CustomWrapper(env, render=False)


# model_name = "model_epoch_2000_low_dim_v15_success_0.5"
# expert_pol = RobomimicExpert(
#     f"models/{model_name}.pth",
#     device="cuda" if torch.cuda.is_available() else "cpu",
# )
# expert_pol.set_env(obs_cacher)


# print(env.env.env.env.env)
# print("\033[32m ref", robomimic_env, "\033[0m")

N = 10000
obs_list, act_list = [], []
ep = 1
while ep <= N:
    ep_obs, ep_act = [], []
    # policy.start_episode()  # important
    # o, done = env.reset(), False
    robomimic_obs, robomimic_done = robomimic_env.reset(), False
    # print(f"raw robosuite: {obs_cacher.latest_obs_dict}")
    # print(f"observation from thrifty dagger: {o}")
    # print(f"observation from reference: {robomimic_obs}")
    # import sys
    # sys.exits()
    step = 0
    while not robomimic_done and len(ep_act) < 300:
        # o_mid_old = get_observation(env=env, di=env.env.observation_spec())
        # a = policy(o_mid_old)  # important
        robomimic_act = policy(robomimic_obs)
        ep_obs.append(deepcopy(_flatten_obs(env.env.observation_spec())))
        ep_act.append(deepcopy(robomimic_act))
        # print(robomimic_obs)
        # print(env.env.observation_spec())
        # print("\033[32m OBS:", ep_obs[-1], "\033[0m")

        # o, _r, _sys_done, _info = env.step(a)
        # done = env._check_success()

        robomimic_obs, _r, _sys_done, _info = robomimic_env.step(robomimic_act)
        robomimic_done = robomimic_env.is_success()['task']
    
    
        # ep_obs.append(robomimic_obs)
        # ep_act.append(robomimic_act)

    
        step += 1
        # print(step)
        # print("\033[32m OBS:" , DeepDiff(get_observation(env, env.env.observation_spec()), robomimic_obs), "\033[0m")
        # print(DeepDiff(a, robomimic_act))
        # print(done == robomimic_done)
    print(f"{ep}: done={robomimic_done}, episode length={len(ep_act)}")
    if robomimic_done:
        ep += 1
        obs_list.extend(ep_obs)
        act_list.extend(ep_act)
        
pickle.dump(
    {"obs": np.array(obs_list), "act": np.array(act_list)},
    open(f"models/{model_name}-{N}.pkl", "wb"),
)

# ep = 1
# while ep <= 300:
#     ep_obs, ep_act = [], []
#     policy.start_episode()
#     o, done = env.reset(), False
#     while not done and len(ep_obs) < 500:
#         a = policy(o)
#         ep_obs.append(o)
#         ep_act.append(a)
#         o, r, sys_done, info = env.step(a)
#         done = env.is_success()["task"]
#     print(f'{ep}: done={done}, episode length={len(ep_obs)}')
#     if done:
#         ep += 1
#         obs_list.extend(ep_obs)
#         act_list.extend(ep_act)
# pickle.dump({"obs": np.array(obs_list), "act": np.array(act_list)}, open(f"models/{model_name}-300.pkl", "wb"))
