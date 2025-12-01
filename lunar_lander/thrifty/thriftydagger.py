from copy import deepcopy
import itertools
import os
import pickle
import random
import time
from typing import Optional

import numpy as np
import torch
from torch.optim import Adam

from . import core
from .utils.logx import EpochLogger


def _reset_env(env, seed=None):
    res = env.reset(seed=seed)
    return res[0] if isinstance(res, tuple) else res


def _step_env(env, action):
    res = env.step(action)
    if len(res) == 5:
        obs, rew, terminated, truncated, info = res
        done = terminated or truncated
    else:
        obs, rew, done, info = res
        truncated = False
    return obs, rew, done, truncated, info


def get_observation_for_thrifty_dagger(obs):
    # LunarLander already returns a flat observation vector
    return obs


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs], act=self.act_buf[idxs])
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }

    def fill_buffer(self, obs, act):
        for i in range(len(obs)):
            self.store(obs[i], act[i])

    def clear(self):
        self.ptr, self.size = 0, 0


class QReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, next_obs, rew, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, pos_fraction=None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }

    def fill_buffer(self, data):
        obs_dim = data["obs"].shape[1]
        for i in range(len(data["obs"]) - 1):
            self.store(
                data["obs"][i],
                data["act"][i],
                data["obs"][i + 1],
                data["rew"][i],
                data["done"][i],
            )
        self.store(
            data["obs"][-1],
            data["act"][-1],
            np.zeros(obs_dim),
            data["rew"][-1],
            data["done"][-1],
        )

    def fill_buffer_from_BC(self, data):
        num_bc = len(data["obs"])
        obs_dim = data["obs"].shape[1]
        for i in range(num_bc - 1):
            self.store(data["obs"][i], data["act"][i], data["obs"][i + 1], 0, 0)
        self.store(
            data["obs"][num_bc - 1], data["act"][num_bc - 1], np.zeros(obs_dim), 1, 1
        )


def generate_offline_data(
    env,
    expert_policy,
    suboptimal_policy=None,
    num_episodes=0,
    output_file="data.pkl",
    seed=0,
):
    i, failures = 0, 0
    np.random.seed(seed)
    obs, act, rew, done = [], [], [], []
    act_limit = env.action_space.high[0]
    while i < num_episodes:
        expert_policy.start_episode()
        if suboptimal_policy:
            suboptimal_policy.start_episode()
        o = _reset_env(env, seed + i)
        total_ret, d, t = 0, False, 0
        while not d:
            a = expert_policy(o)
            if a is None:
                d, r = True, 0
                continue
            a = np.clip(a, -act_limit, act_limit)
            obs.append(o)
            act.append(a)
            o, r, d, truncated, _ = _step_env(env, a)
            d = d or truncated
            total_ret += r
            rew.append(r)
            done.append(d)
            t += 1
        i += 1
        if total_ret > 0:
            print(f"Collected episode #{i} return {total_ret:.2f} length {t}")
        else:
            failures += 1
            print(f"Episode #{i} failed with return {total_ret:.2f}")
    print("Ep Mean, Std Dev:", np.array(rew).mean(), np.array(rew).std())
    print("Num Successes {} Num Failures {}".format(num_episodes - failures, failures))
    pickle.dump(
        {"obs": np.stack(obs), "act": np.stack(act), "rew": np.array(rew), "done": np.array(done)},
        open(output_file, "wb"),
    )


def thrifty(
    env,
    iters=5,
    actor_critic=core.Ensemble,
    ac_kwargs=dict(),
    seed=0,
    grad_steps=500,
    obs_per_iter=2000,
    replay_size=int(3e4),
    pi_lr=1e-3,
    batch_size=100,
    logger_kwargs=dict(),
    num_test_episodes=10,
    bc_epochs=5,
    input_file="data.pkl",
    device_idx=0,
    expert_policy=None,
    suboptimal_policy=None,
    num_nets=5,
    target_rate=0.01,
    success_threshold=200.0,
    q_learning=False,
    gamma=0.99,
    init_model=None,
    max_expert_query=2000,
):
    logger = EpochLogger(**logger_kwargs)
    _locals = locals()
    del _locals["env"]
    try:
        logger.save_config(_locals)
    except Exception:
        pass

    device = torch.device("cuda", device_idx) if device_idx >= 0 else torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    assert act_limit == -1 * env.action_space.low[0], "Action space should be symmetric"
    horizon = getattr(env, "_max_episode_steps", None)
    if horizon is None and env.spec is not None:
        horizon = env.spec.max_episode_steps
    horizon = horizon or obs_per_iter

    ac = actor_critic(env.observation_space, env.action_space, device, num_nets=num_nets, **ac_kwargs)
    if init_model:
        try:
            ac = torch.load(init_model, map_location=device, weights_only=False).to(device)
        except TypeError:
            ac = torch.load(init_model, map_location=device).to(device)
        ac.device = device
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False
    pi_optimizers = [Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)]
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=pi_lr)
    logger.setup_pytorch_saver(ac)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    input_data = pickle.load(open(input_file, "rb"))
    num_bc = len(input_data["obs"])
    idxs = np.arange(num_bc)
    np.random.shuffle(idxs)
    replay_buffer.fill_buffer(
        input_data["obs"][idxs][: int(0.9 * num_bc)],
        input_data["act"][idxs][: int(0.9 * num_bc)],
    )
    held_out_data = {
        "obs": input_data["obs"][idxs][int(0.9 * num_bc) :],
        "act": input_data["act"][idxs][int(0.9 * num_bc) :],
    }
    qbuffer = QReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    qbuffer.fill_buffer_from_BC(input_data)

    def compute_loss_pi(data, i):
        o, a = data["obs"], data["act"]
        a_pred = ac.pis[i](o)
        return torch.mean(torch.sum((a - a_pred) ** 2, dim=1))

    def compute_loss_q(data):
        o, a, o2, r, d = data["obs"], data["act"], data["obs2"], data["rew"], data["done"]
        with torch.no_grad():
            a2 = torch.mean(torch.stack([pi(o2) for pi in ac.pis]), dim=0)
        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        with torch.no_grad():
            q1_t = ac_targ.q1(o2, a2)
            q2_t = ac_targ.q2(o2, a2)
            backup = r + gamma * (1 - d) * torch.min(q1_t, q2_t)
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        return loss_q1 + loss_q2

    def update_pi(data, i):
        pi_optimizers[i].zero_grad()
        loss_pi = compute_loss_pi(data, i)
        loss_pi.backward()
        pi_optimizers[i].step()
        return loss_pi.item()

    def update_q(data, timer):
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        if timer % 2 == 0:
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(0.995)
                    p_targ.data.add_((1 - 0.995) * p.data)
        return loss_q.item()

    if iters == 0 and num_test_episodes > 0:
        _ = _run_policy_rollouts(env, ac, num_test_episodes, act_limit, success_threshold)
        return

    for i_net in range(ac.num_nets):
        tmp_buffer = replay_buffer if ac.num_nets == 1 else ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
        )
        if ac.num_nets > 1:
            for _ in range(replay_buffer.size):
                idx = np.random.randint(replay_buffer.size)
                tmp_buffer.store(replay_buffer.obs_buf[idx], replay_buffer.act_buf[idx])
        for _ in range(bc_epochs):
            for _ in range(grad_steps):
                batch = tmp_buffer.sample_batch(batch_size)
                update_pi(batch, i_net)

    discrepancies, estimates = [], []
    for i in range(replay_buffer.size):
        a_pred = ac.act(replay_buffer.obs_buf[i])
        a_sup = replay_buffer.act_buf[i]
        discrepancies.append(sum((a_pred - a_sup) ** 2))
        estimates.append(ac.variance(replay_buffer.obs_buf[i]))
    heldout_estimates = [ac.variance(o) for o in held_out_data["obs"]]
    switch2robot_thresh = np.array(discrepancies).mean()
    target_idx = int((1 - target_rate) * len(heldout_estimates))
    switch2human_thresh = sorted(heldout_estimates)[target_idx]
    switch2human_thresh2 = 0.48
    switch2robot_thresh2 = 0.495
    torch.cuda.empty_cache()
    replay_buffer.fill_buffer(held_out_data["obs"], held_out_data["act"])

    total_env_interacts, ep_num, fail_ct = 0, 0, 0
    online_burden, num_switch_to_human, num_switch_to_human2, num_switch_to_robot = 0, 0, 0, 0

    for t in range(iters + 1):
        logging_data = []
        estimates = []
        estimates2 = []
        i = 0
        if t == 0:
            i = obs_per_iter
        while i < obs_per_iter:
            expert_policy.start_episode()
            if suboptimal_policy:
                suboptimal_policy.start_episode()
            o = _reset_env(env, seed + t + i)
            expert_mode, safety_mode, ep_len = False, False, 0
            obs_traj, act_traj, rew_traj, done_traj, sup_traj, var_traj, risk_traj = (
                [o],
                [],
                [],
                [],
                [],
                [ac.variance(o)],
                [],
            )
            ep_return = 0.0
            while i < obs_per_iter and ep_len < horizon:
                a = ac.act(o)
                a = np.clip(a, -act_limit, act_limit)
                a_expert = expert_policy(o)
                if not expert_mode:
                    estimates.append(ac.variance(o))
                    estimates2.append(ac.safety(o, a))
                if expert_mode:
                    a_env = np.clip(a_expert, -act_limit, act_limit)
                    replay_buffer.store(o, a_env)
                    online_burden += 1
                    risk_traj.append(ac.safety(o, a_env))
                    if (np.sum((a - a_env) ** 2) < switch2robot_thresh) and (
                        not q_learning or ac.safety(o, a) > switch2robot_thresh2
                    ):
                        expert_mode = False
                        num_switch_to_robot += 1
                    o2, r, d, truncated, _ = _step_env(env, a_env)
                    success_flag = d and (ep_return + r >= success_threshold)
                    qbuffer.store(o, a_env, o2, int(success_flag), d or truncated)
                    act_traj.append(a_env)
                    sup_traj.append(1)
                elif (
                    not expert_mode and ac.variance(o) > switch2human_thresh
                ):
                    expert_mode = True
                    num_switch_to_human += 1
                    continue
                elif (
                    not expert_mode
                    and q_learning
                    and ac.safety(o, a) < switch2human_thresh2
                ):
                    safety_mode = True
                    num_switch_to_human2 += 1
                    continue
                else:
                    a_env = a
                    o2, r, d, truncated, _ = _step_env(env, a_env)
                    success_flag = d and (ep_return + r >= success_threshold)
                    qbuffer.store(o, a_env, o2, int(success_flag), d or truncated)
                    act_traj.append(a_env)
                    sup_traj.append(0)
                    risk_traj.append(ac.safety(o, a_env))
                ep_return += r
                d = d or truncated
                rew_traj.append(r)
                done_traj.append(d)
                o = get_observation_for_thrifty_dagger(o2)
                obs_traj.append(o)
                var_traj.append(ac.variance(o))
                i += 1
                ep_len += 1
                if d:
                    break
            ep_num += 1
            if ep_return < success_threshold:
                fail_ct += 1
            total_env_interacts += ep_len
            logging_data.append(
                {
                    "obs": np.stack(obs_traj),
                    "act": np.stack(act_traj),
                    "done": np.array(done_traj),
                    "rew": np.array(rew_traj),
                    "sup": np.array(sup_traj),
                    "var": np.array(var_traj),
                    "risk": np.array(risk_traj),
                    "beta_H": switch2human_thresh,
                    "beta_R": switch2robot_thresh,
                    "eps_H": switch2human_thresh2,
                    "eps_R": switch2robot_thresh2,
                }
            )
            pickle.dump(logging_data, open(os.path.join(logger_kwargs["output_dir"], f"iter{t}.pkl"), "wb"))
            if len(estimates) > 25:
                target_idx = int((1 - target_rate) * len(estimates))
                switch2human_thresh = sorted(estimates)[target_idx]
                switch2human_thresh2 = sorted(estimates2, reverse=True)[target_idx]
                switch2robot_thresh2 = sorted(estimates2)[int(0.5 * len(estimates))]

        if t > 0:
            loss_pi = []
            ac = actor_critic(
                env.observation_space,
                env.action_space,
                device,
                num_nets=num_nets,
                **ac_kwargs,
            )
            pi_optimizers = [Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)]
            for i_net in range(ac.num_nets):
                tmp_buffer = replay_buffer if ac.num_nets == 1 else ReplayBuffer(
                    obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
                )
                if ac.num_nets > 1:
                    for _ in range(replay_buffer.size):
                        idx = np.random.randint(replay_buffer.size)
                        tmp_buffer.store(replay_buffer.obs_buf[idx], replay_buffer.act_buf[idx])
                for _ in range(grad_steps * (bc_epochs + t)):
                    batch = tmp_buffer.sample_batch(batch_size)
                    loss_pi.append(update_pi(batch, i_net))
        if q_learning:
            loss_q = []
            for _ in range(bc_epochs):
                for i_upd in range(grad_steps * 5):
                    batch = qbuffer.sample_batch(batch_size // 2)
                    loss_q.append(update_q(batch, timer=i_upd))
        logger.save_state(dict())
        success_rate = (ep_num - fail_ct) / ep_num if ep_num > 0 else 0.0
        logger.log_tabular("Epoch", t)
        logger.log_tabular("TotalEpisodes", ep_num)
        logger.log_tabular("TotalSuccesses", ep_num - fail_ct)
        logger.log_tabular("SuccessRate", success_rate)
        logger.log_tabular("TotalEnvInteracts", total_env_interacts)
        logger.log_tabular("OnlineBurden", online_burden)
        logger.log_tabular("NumSwitchToNov", num_switch_to_human)
        logger.log_tabular("NumSwitchToRisk", num_switch_to_human2)
        logger.log_tabular("NumSwitchBack", num_switch_to_robot)
        logger.dump_tabular()
        if online_burden >= max_expert_query:
            print("Reached max expert queries, stopping training.")
            break


def _run_policy_rollouts(env, ac, num_episodes, act_limit, success_threshold):
    returns = []
    for _ in range(num_episodes):
        o = _reset_env(env)
        ep_ret, done = 0.0, False
        while not done:
            a = np.clip(ac.act(o), -act_limit, act_limit)
            o, r, done, truncated, _ = _step_env(env, a)
            done = done or truncated
            ep_ret += r
        returns.append(ep_ret)
    success_rate = np.mean([ret >= success_threshold for ret in returns])
    print(f"Test Success Rate: {success_rate:.2f} over {num_episodes} episodes")
    return returns
