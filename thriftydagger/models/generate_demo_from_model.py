import gymnasium as gym
import pickle, numpy as np, torch
from robomimic.algo import algo_factory
from robomimic.utils.file_utils import env_from_checkpoint, policy_from_checkpoint

model_name = "model_epoch_2000_low_dim_v15_success_0.5"
ckpt = f"models/{model_name}.pth"
env, ckpt_dict = env_from_checkpoint(ckpt_path=ckpt, render=False)
policy, _ = policy_from_checkpoint(ckpt_dict=ckpt_dict)
print(policy)
obs_list, act_list = [], []
ep = 1
while ep <= 30:
    ep_obs, ep_act = [], []
    env.reset()
    policy.start_episode()
    o, done = env.reset(), False
    while not done and len(ep_obs) < 10000:
        a = policy(o)
        ep_obs.append(o)
        ep_act.append(a)
        o, r, done, info = env.step(a)
    print(f'{ep}: done={done}, episode length={len(ep_obs)}')
    if done:
        ep += 1
        obs_list.extend(ep_obs)
        act_list.extend(ep_act)
pickle.dump({"obs": np.array(obs_list), "act": np.array(act_list)}, open(f"models/{model_name}.pkl", "wb"))