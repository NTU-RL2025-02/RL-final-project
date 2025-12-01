"""
thriftydagger.py: Thrifty DAgger algorithm implementation
"""

import os
import sys
import pickle
import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam

import thrifty.algos.core as core
from thrifty.utils.logx import EpochLogger
from thrifty.algos.buffer import ReplayBuffer, QReplayBuffer


def get_observation_for_thrifty_dagger(env):
    obs_dict = env.env.observation_spec()
    return np.concat([obs_dict["robot0_proprio-state"], obs_dict["object-state"]])


# ----------------------------------------------------------------------
# Config dataclasses（集中 magic numbers）
# ----------------------------------------------------------------------


@dataclass
class ThresholdConfig:
    """控制 switching 門檻更新的相關常數。"""

    # online estimates 數量大於這個值才更新門檻
    min_estimates_for_update: int = 25
    # Q-risk 初始切到 human 的 safety 門檻（折扣成功率）
    init_eps_H: float = 0.48
    # Q-risk 初始切回 robot 的 safety 門檻
    init_eps_R: float = 0.495


@dataclass
class QRiskConfig:
    """控制 Q-risk 訓練相關的超參數。"""

    # 每個 batch 中 positive（成功）樣本佔的比例
    pos_fraction: float = 0.1
    # Q-network 的 gradient step 是 policy 的幾倍
    q_grad_multiplier: int = 5
    # Q-network 的 batch 大小 = batch_size * q_batch_scale
    q_batch_scale: float = 0.5


# ----------------------------------------------------------------------
# Offline data collection
# ----------------------------------------------------------------------


def generate_offline_data(
    env: Any,
    expert_policy: Any,
    recovery_policy: Optional[Any] = None,
    num_episodes: int = 0,
    output_file: str = "data.pkl",
    robosuite: bool = False,
    robosuite_cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> None:
    """
    使用 expert_policy 在 env 上 roll out 若干 episodes，
    收集 (obs, act) 並存成離線 BC 資料（pickle: {"obs", "act"}）。

    若 robosuite=True，僅保留成功 episodes 的資料。
    """
    episode_idx, failures = 0, 0
    np.random.seed(seed)
    obs, act, returns = [], [], []

    # act_limit: action 空間的上界 (假設是對稱的)
    act_limit = env.action_space.high[0]

    while episode_idx < num_episodes:
        print(f"Episode #{episode_idx}")
        # 通知 expert / recovery policy 開始新的 episode
        expert_policy.start_episode()
        if recovery_policy is not None:
            recovery_policy.start_episode()  # NOTE: 為啥會有這？ by AaW

        o, total_ret, done, step_idx = env.reset(), 0.0, False, 0
        curr_obs, curr_act = [], []

        # if robosuite:
        #     robosuite_cfg["INPUT_DEVICE"].start_control()

        while not done:
            # 由 expert_policy 給出 action
            a = expert_policy(o)
            if a is None:
                # 若 expert 回傳 None，視為無法操作，直接終止
                done, r = True, 0.0
                continue

            a = np.clip(a, -act_limit, act_limit)

            curr_obs.append(o)
            curr_act.append(a)

            # 在環境中執行 action
            o, r, done, _ = env.step(a)
            if robosuite:
                # robosuite 的 done / reward 由 _check_success() 控制
                done = (step_idx >= robosuite_cfg["MAX_EP_LEN"]) or env._check_success()
                r = float(env._check_success())

            total_ret += r
            step_idx += 1

        if robosuite:
            # robosuite 版本只保留成功 episodes
            if total_ret > 0:
                episode_idx += 1
                obs.extend(curr_obs)
                act.extend(curr_act)
            else:
                failures += 1
            env.close()
        else:
            episode_idx += 1
            obs.extend(curr_obs)
            act.extend(curr_act)

        print(f"Collected episode with return {total_ret} length {step_idx}")
        returns.append(total_ret)

    print("Ep Mean, Std Dev:", np.mean(returns), np.std(returns))
    print(f"Num Successes {num_episodes} Num Failures {failures}")

    # 將 (obs, act) 存成 pickle 檔供之後 BC 使用
    pickle.dump({"obs": np.stack(obs), "act": np.stack(act)}, open(output_file, "wb"))


# ----------------------------------------------------------------------
# Losses & single-step updates
# ----------------------------------------------------------------------


def compute_loss_pi(
    ac: Any, data: Dict[str, torch.Tensor], net_idx: int
) -> torch.Tensor:
    """對 ensemble 中第 net_idx 個 policy 做 MSE 行為克隆 loss。"""
    o, a = data["obs"], data["act"]
    a_pred = ac.pis[net_idx](o)
    return torch.mean(torch.sum((a - a_pred) ** 2, dim=1))


def compute_loss_q(
    ac: Any, ac_targ: Any, data: Dict[str, torch.Tensor], gamma: float
) -> torch.Tensor:
    """
    雙 Q-network 的 MSE Bellman loss：
      backup = r + gamma * (1 - d) * min(Q1', Q2').
    """
    o, a, o2, r, d = (
        data["obs"],
        data["act"],
        data["obs2"],
        data["rew"],
        data["done"],
    )

    with torch.no_grad():
        # ensemble 平均的 a2
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


def update_pi(
    ac: Any, pi_optimizer: Adam, data: Dict[str, torch.Tensor], net_idx: int
) -> float:
    """對指定的 ensemble policy 做一次 gradient step，回傳 loss (float)。"""
    pi_optimizer.zero_grad()
    loss_pi = compute_loss_pi(ac, data, net_idx)
    loss_pi.backward()
    pi_optimizer.step()
    return float(loss_pi.item())


def update_q(
    ac: Any,
    ac_targ: Any,
    q_optimizer: Adam,
    data: Dict[str, torch.Tensor],
    gamma: float,
    timer: int,
) -> float:
    """對 Q-network 做一次 gradient step，包含 soft update target network。"""
    q_optimizer.zero_grad()
    loss_q = compute_loss_q(ac, ac_targ, data, gamma)
    loss_q.backward()
    q_optimizer.step()

    if timer % 2 == 0:
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(0.995)
                p_targ.data.add_((1 - 0.995) * p.data)

    return float(loss_q.item())


# ----------------------------------------------------------------------
# Evaluation rollouts
# ----------------------------------------------------------------------


def test_agent(
    expert_policy: Any,
    recovery_policy: Any,
    env: Any,
    env_robomimic: Any,
    ac: Any,
    num_test_episodes: int,
    act_limit: float,
    horizon: int,
    robosuite: bool,
    logger_kwargs: Dict[str, Any],
    epoch: int = 0,
) -> None:
    """
    使用目前 policy `ac` 在環境中跑 `num_test_episodes` 回合（不做 intervention），
    並將 rollouts 存成 `test-rollouts.pkl` 以及 `output_dir/test{epoch}.pkl`。
    """
    obs, act, done, rew = [], [], [], []

    for episode_idx in range(num_test_episodes):
        expert_policy.start_episode()
        recovery_policy.start_episode()

        ep_ret, ep_ret2, ep_len = 0.0, 0.0, 0
        o_robomimic, terminated = env_robomimic.reset(), False
        o = get_observation_for_thrifty_dagger(env)

        while not terminated:
            obs.append(o)
            a = ac.act(o)
            a = np.clip(a, -act_limit, act_limit)
            act.append(a)

            o_robomimic, r, terminated, _ = env_robomimic.step(a)
            o = get_observation_for_thrifty_dagger(env)

            if robosuite:
                terminated = (ep_len + 1 >= horizon) or env._check_success()
                ep_ret2 += float(env._check_success())
                done.append(terminated)
                rew.append(int(env._check_success()))

            ep_ret += r
            ep_len += 1

        print(f"episode #{episode_idx} success? {rew[-1]}")
        if robosuite:
            env.close()

    print("Test Success Rate:", sum(rew) / num_test_episodes)

    data = {
        "obs": np.stack(obs),
        "act": np.stack(act),
        "done": np.array(done),
        "rew": np.array(rew),
    }

    pickle.dump(data, open("test-rollouts.pkl", "wb"))
    pickle.dump(
        data,
        open(os.path.join(logger_kwargs["output_dir"], f"test{epoch}.pkl"), "wb"),
    )


# ----------------------------------------------------------------------
# Offline BC pretraining
# ----------------------------------------------------------------------


def pretrain_policies(
    ac: Any,
    replay_buffer: ReplayBuffer,
    held_out_data: Dict[str, np.ndarray],
    grad_steps: int,
    bc_epochs: int,
    batch_size: int,
    replay_size: int,
    obs_dim: Tuple[int, ...],
    act_dim: int,
    device: torch.device,
    pi_lr: float,
) -> List[Adam]:
    """
    使用離線 BC 資料先 pretrain ensemble 中的每個 policy。

    每個成員都各自抽樣（bootstrap）一個 dataset，並在 held-out 上做簡單 validation。
    """
    pi_optimizers: List[Adam] = [
        Adam(ac.pis[net_idx].parameters(), lr=pi_lr) for net_idx in range(ac.num_nets)
    ]

    for net_idx in range(ac.num_nets):
        if ac.num_nets > 1:
            print(f"Net #{net_idx}")
            tmp_buffer = ReplayBuffer(
                obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
            )
            for _ in range(replay_buffer.size):
                buf_idx = np.random.randint(replay_buffer.size)
                tmp_buffer.store(
                    replay_buffer.obs_buf[buf_idx], replay_buffer.act_buf[buf_idx]
                )
        else:
            tmp_buffer = replay_buffer

        for epoch_idx in range(bc_epochs):
            loss_pi_vals: List[float] = []

            for step_idx in range(grad_steps):
                batch = tmp_buffer.sample_batch(batch_size)
                loss_pi_vals.append(
                    update_pi(ac, pi_optimizers[net_idx], batch, net_idx)
                )

            validation_losses: List[float] = []
            for sample_idx in range(len(held_out_data["obs"])):
                a_pred = ac.act(held_out_data["obs"][sample_idx], i=net_idx)
                a_sup = held_out_data["act"][sample_idx]
                validation_losses.append(float(np.sum(a_pred - a_sup) ** 2))

            print("LossPi", sum(loss_pi_vals) / len(loss_pi_vals))
            print("LossValid", sum(validation_losses) / len(validation_losses))

    return pi_optimizers


# ----------------------------------------------------------------------
# Threshold estimation from offline data
# ----------------------------------------------------------------------


def estimate_initial_thresholds(
    ac: Any,
    replay_buffer: ReplayBuffer,
    held_out_data: Dict[str, np.ndarray],
    target_rate: float,
) -> Tuple[float, float]:
    """
    從 offline BC data 估計：
      - switch2robot_thresh: robot vs expert discrepancy 的平均
      - switch2human_thresh: ensemble variance 在 held-out set 上的 (1 - target_rate) 分位數
    """
    discrepancies: List[float] = []
    for buf_idx in range(replay_buffer.size):
        a_pred = ac.act(replay_buffer.obs_buf[buf_idx])
        a_sup = replay_buffer.act_buf[buf_idx]
        discrepancies.append(float(np.sum((a_pred - a_sup) ** 2)))

    heldout_estimates: List[float] = []
    for sample_idx in range(len(held_out_data["obs"])):
        heldout_estimates.append(float(ac.variance(held_out_data["obs"][sample_idx])))

    switch2robot_thresh = float(np.mean(discrepancies))

    # 取 (1 - target_rate) 分位數
    target_idx = int((1 - target_rate) * len(heldout_estimates))
    switch2human_thresh = sorted(heldout_estimates)[target_idx]

    return switch2robot_thresh, switch2human_thresh


# ----------------------------------------------------------------------
# Helpers: retrain policy / Q-risk / logging
# ----------------------------------------------------------------------


def retrain_policy(
    actor_critic_cls: Any,
    env: Any,
    device: torch.device,
    num_nets: int,
    ac_kwargs: Dict[str, Any],
    replay_buffer: ReplayBuffer,
    replay_size: int,
    obs_dim: Tuple[int, ...],
    act_dim: int,
    pi_lr: float,
    grad_steps: int,
    bc_epochs: int,
    epoch_idx: int,
    batch_size: int,
) -> Tuple[Any, List[Adam], Optional[float]]:
    """
    使用 aggregate replay_buffer 的資料，從頭重新訓練 ensemble policy。
    回傳新的 ac、pi_optimizers 與平均 LossPi。
    """
    if epoch_idx == 0:
        return None, [], None  # epoch 0 不 retrain

    loss_pi_vals: List[float] = []

    ac = actor_critic_cls(
        env.observation_space,
        env.action_space,
        device,
        num_nets=num_nets,
        **ac_kwargs,
    )

    pi_optimizers: List[Adam] = [
        Adam(ac.pis[net_idx].parameters(), lr=pi_lr) for net_idx in range(ac.num_nets)
    ]

    for net_idx in range(ac.num_nets):
        # bootstrap resampling，讓每個 ensemble 成員看到不同的 dataset
        if ac.num_nets > 1:
            tmp_buffer = ReplayBuffer(
                obs_dim=obs_dim,
                act_dim=act_dim,
                size=replay_size,
                device=device,
            )
            for _ in range(replay_buffer.size):
                buf_idx = np.random.randint(replay_buffer.size)
                tmp_buffer.store(
                    replay_buffer.obs_buf[buf_idx], replay_buffer.act_buf[buf_idx]
                )
        else:
            tmp_buffer = replay_buffer

        # 訓練次數會隨著 epoch 增加
        total_steps = grad_steps * (bc_epochs + epoch_idx)
        for step_idx in range(total_steps):
            batch = tmp_buffer.sample_batch(batch_size)
            loss_pi_vals.append(update_pi(ac, pi_optimizers[net_idx], batch, net_idx))

    avg_loss_pi = sum(loss_pi_vals) / len(loss_pi_vals) if loss_pi_vals else None
    return ac, pi_optimizers, avg_loss_pi


def retrain_qrisk(
    ac: Any,
    ac_targ: Any,
    qbuffer: QReplayBuffer,
    q_learning: bool,
    num_test_episodes: int,
    expert_policy: Any,
    recovery_policy: Any,
    env: Any,
    env_robomimic: Any,
    act_limit: float,
    horizon: int,
    robosuite: bool,
    logger_kwargs: Dict[str, Any],
    pi_lr: float,
    bc_epochs: int,
    grad_steps: int,
    gamma: float,
    batch_size: int,
    qrisk_cfg: QRiskConfig,
    epoch_idx: int,
) -> Optional[float]:
    """
    若 q_learning=True，重新訓練 Q-risk safety critic，並回傳平均 LossQ。
    否則回傳 None。
    """
    if not q_learning:
        return None

    # 用目前 policy 收集離線資料（optional）
    if num_test_episodes > 0:
        test_agent(
            expert_policy,
            recovery_policy,
            env,
            env_robomimic,
            ac,
            num_test_episodes,
            act_limit,
            horizon,
            robosuite,
            logger_kwargs,
            epoch=epoch_idx,
        )
        data = pickle.load(open("test-rollouts.pkl", "rb"))
        qbuffer.fill_buffer(data)
        os.remove("test-rollouts.pkl")

    # 重新設定 Q-network optimizer
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=pi_lr)

    loss_q_vals: List[float] = []
    q_batch_size = int(batch_size * qrisk_cfg.q_batch_scale)

    for _ in range(bc_epochs):
        for step_idx in range(grad_steps * qrisk_cfg.q_grad_multiplier):
            batch = qbuffer.sample_batch(
                q_batch_size, pos_fraction=qrisk_cfg.pos_fraction
            )
            loss_q_vals.append(
                update_q(ac, ac_targ, q_optimizer, batch, gamma, timer=step_idx)
            )

    avg_loss_q = sum(loss_q_vals) / len(loss_q_vals) if loss_q_vals else None
    return avg_loss_q


def log_epoch(
    logger: EpochLogger,
    epoch_idx: int,
    ep_num: int,
    fail_ct: int,
    total_env_interacts: int,
    online_burden: int,
    num_switch_to_human: int,
    num_switch_to_recovery: int,
    num_switch_to_robot: int,
    loss_pi: Optional[float],
    loss_q: Optional[float],
    q_learning: bool,
) -> None:
    """
    負責：
      - 呼叫 logger.save_state（外部呼叫）
      - 印出當前統計
      - 使用 logger.log_tabular 寫入 progress.txt
    """
    print("Epoch", epoch_idx)
    avg_loss_pi = loss_pi if loss_pi is not None else 0.0
    avg_loss_q = loss_q if loss_q is not None else 0.0

    if loss_pi is not None:
        print("LossPi", avg_loss_pi)
    if q_learning and loss_q is not None:
        print("LossQ", avg_loss_q)

    print("TotalEpisodes", ep_num)
    print("TotalSuccesses", ep_num - fail_ct)
    print("TotalEnvInteracts", total_env_interacts)
    print("OnlineBurden", online_burden)
    print("NumSwitchToNov", num_switch_to_human)
    print("NumSwitchToRisk", num_switch_to_recovery)
    print("NumSwitchBack", num_switch_to_robot)

    success_rate = (ep_num - fail_ct) / ep_num if ep_num > 0 else 0.0

    logger.log_tabular("Epoch", epoch_idx)
    logger.log_tabular("LossPi", avg_loss_pi)
    if q_learning:
        logger.log_tabular("LossQ", avg_loss_q)
    logger.log_tabular("TotalEpisodes", ep_num)
    logger.log_tabular("TotalSuccesses", ep_num - fail_ct)
    logger.log_tabular("SuccessRate", success_rate)
    logger.log_tabular("TotalEnvInteracts", total_env_interacts)
    logger.log_tabular("OnlineBurden", online_burden)
    logger.log_tabular("NumSwitchToNov", num_switch_to_human)
    logger.log_tabular("NumSwitchToRisk", num_switch_to_recovery)
    logger.log_tabular("NumSwitchBack", num_switch_to_robot)

    logger.dump_tabular()


# ----------------------------------------------------------------------
# Main ThriftyDAgger algorithm
# ----------------------------------------------------------------------


def thrifty(
    env: Any,
    env_robomimic: Any,
    iters: int = 5,
    actor_critic: Any = core.Ensemble,
    ac_kwargs: Dict[str, Any] = dict(),
    seed: int = 0,
    grad_steps: int = 500,
    obs_per_iter: int = 2000,
    replay_size: int = int(3e4),
    pi_lr: float = 1e-3,
    batch_size: int = 100,
    logger_kwargs: Dict[str, Any] = dict(),
    num_test_episodes: int = 10,
    bc_epochs: int = 20,
    input_file: str = "data.pkl",
    device_idx: int = 0,
    expert_policy: Optional[Any] = None,
    recovery_policy: Optional[Any] = None,
    extra_obs_extractor: Optional[Any] = None,
    num_nets: int = 5,
    target_rate: float = 0.01,
    robosuite: bool = False,
    robosuite_cfg: Optional[Dict[str, Any]] = None,
    q_learning: bool = False,
    gamma: float = 0.9999,
    init_model: Optional[str] = None,
    max_expert_query: int = 2000,
) -> None:
    """
    Main entrypoint for ThriftyDAgger.

    主要流程：
      1. 使用 offline BC data pretrain ensemble policies
      2. 根據線上 estimate 的 uncertainty / Q-risk 做 context switching
      3. 每個 iter 重新用 aggregate data 做一次 DAgger-style retrain
      4. 選擇性地訓練 Q-risk safety critic
    """

    # ----------------------------------------------------------
    # 1. 建立 logger 並存 config（不包含 env，避免 JSON 序列化問題）
    # ----------------------------------------------------------
    logger = EpochLogger(**logger_kwargs)
    _locals = locals()
    del _locals["env"]  # env 無法 JSON 化，從 config 中移除
    try:
        logger.save_config(_locals)
    except TypeError as e:
        print(f"[Warning] Could not save config as JSON: {e}")

    # ----------------------------------------------------------
    # 2. 裝置選擇與隨機種子
    # ----------------------------------------------------------
    if device_idx >= 0 and torch.cuda.is_available():
        device = torch.device("cuda", device_idx)
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ----------------------------------------------------------
    # 3. robosuite 設定與環境基本資訊
    # ----------------------------------------------------------
    if robosuite:
        # 將 model.xml 存到 output_dir，方便之後 replay
        with open(os.path.join(logger_kwargs["output_dir"], "model.xml"), "w") as fh:
            fh.write(env.env.sim.model.get_xml())

    # 從環境取得 observation space / action space 維度
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # 確保 action space 是對稱的 [-act_limit, act_limit]
    assert act_limit == -1 * env.action_space.low[0], "Action space should be symmetric"

    # horizon: 每個 episode 最大長度 (由 robosuite_cfg 設定)
    horizon = robosuite_cfg["MAX_EP_LEN"]

    # ----------------------------------------------------------
    # 4. 建立 actor-critic（ensemble）與 target network
    # ----------------------------------------------------------
    ac = actor_critic(
        env.observation_space,
        env.action_space,
        device,
        num_nets=num_nets,
        **ac_kwargs,
    )

    if init_model:
        # 若提供 init_model，則直接從檔案載入已訓練好的 ac
        try:
            ac = torch.load(init_model, map_location=device, weights_only=False).to(
                device
            )
        except TypeError:
            ac = torch.load(init_model, map_location=device).to(device)
        ac.device = device

    # target network（Q-learning 用）
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # ----------------------------------------------------------
    # 5. Q-network optimizer（第一次初始化）與 model saving 設定
    #    （實際訓練時會在 retrain_qrisk 中重新建立 optimizer）
    # ----------------------------------------------------------
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=pi_lr)  # 主要是占位，實際訓練時重建
    logger.setup_pytorch_saver(ac)

    # ----------------------------------------------------------
    # 6. 建立 replay buffers 並載入 offline data
    # ----------------------------------------------------------
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=replay_size,
        device=device,
    )

    # 載入離線 expert rollouts (pkl file)
    input_data = pickle.load(open(input_file, "rb"))

    # shuffle 並切出 held-out set 用於 validation / threshold estimation
    num_bc = len(input_data["obs"])
    idxs = np.arange(num_bc)
    np.random.shuffle(idxs)

    # 前 90% 當作 BC 訓練資料
    replay_buffer.fill_buffer(
        input_data["obs"][idxs][: int(0.9 * num_bc)],
        input_data["act"][idxs][: int(0.9 * num_bc)],
    )

    # 後 10% 當作 held-out set
    held_out_data = {
        "obs": input_data["obs"][idxs][int(0.9 * num_bc) :],
        "act": input_data["act"][idxs][int(0.9 * num_bc) :],
    }

    # Q-replay buffer，用來訓練 safety critic (Qrisk)
    qbuffer = QReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=replay_size,
        device=device,
    )
    # 從 BC 的離線資料初始化 Q buffer
    qbuffer.fill_buffer_from_BC(input_data)

    # ----------------------------------------------------------
    # 7. 純 evaluation 模式（iters=0 且有設定 test episodes）
    # ----------------------------------------------------------
    if iters == 0 and num_test_episodes > 0:
        test_agent(
            expert_policy,
            recovery_policy,
            env,
            env_robomimic,
            ac,
            num_test_episodes,
            act_limit,
            horizon,
            robosuite,
            logger_kwargs,
            epoch=0,
        )
        sys.exit(0)

    # ----------------------------------------------------------
    # 8. 利用 offline data 先做 pre-training (BC)
    # ----------------------------------------------------------
    pi_optimizers = pretrain_policies(
        ac,
        replay_buffer,
        held_out_data,
        grad_steps,
        bc_epochs,
        batch_size,
        replay_size,
        obs_dim,
        act_dim,
        device,
        pi_lr,
    )

    # ----------------------------------------------------------
    # 9. 初始化統計量與 thresholds
    # ----------------------------------------------------------
    switch2robot_thresh, switch2human_thresh = estimate_initial_thresholds(
        ac, replay_buffer, held_out_data, target_rate
    )
    print("Estimated switch-back threshold:", switch2robot_thresh)
    print("Estimated switch-to threshold:", switch2human_thresh)

    threshold_cfg = ThresholdConfig()
    qrisk_cfg = QRiskConfig()

    switch2human_thresh2 = threshold_cfg.init_eps_H
    switch2robot_thresh2 = threshold_cfg.init_eps_R

    torch.cuda.empty_cache()
    replay_buffer.fill_buffer(held_out_data["obs"], held_out_data["act"])

    # 訓練過程中統計資訊
    total_env_interacts = 0  # 環境互動的總步數
    ep_num = 0  # 總 episode 數
    fail_ct = 0  # 失敗 episode 數（超過 horizon）
    online_burden = 0  # supervisor 標註總數
    num_switch_to_human = 0  # 因 novelty 切到 human 次數
    num_switch_to_recovery = 0  # 因 risk 切到 human 次數
    num_switch_to_robot = 0  # 從 human/recovery 切回 robot 次數

    # ----------------------------------------------------------
    # 10. Main ThriftyDAgger Loop
    # ----------------------------------------------------------
    for epoch_idx in range(iters + 1):
        # --------------------------------------------------
        # 10-1. 線上資料收集（epoch 0 跳過，保留給純 Q-training）
        # --------------------------------------------------
        step_count = 0
        if epoch_idx == 0:
            step_count = obs_per_iter  # 不跑 while loop

        logging_data: List[Dict[str, Any]] = []
        estimates: List[float] = []
        estimates2: List[float] = []

        while step_count < obs_per_iter:
            expert_policy.start_episode()
            recovery_policy.start_episode()

            o_robomimic, done = env_robomimic.reset(), False
            o = get_observation_for_thrifty_dagger(env)

            expert_mode = False
            safety_mode = False
            ep_len = 0

            # episode 累積的軌跡
            obs, act, rew, done_flags, sup, var, risk = (
                [o],  # 初始 obs
                [],  # actions
                [],  # rewards（成功:1 / 失敗:0）
                [],  # done flags
                [],  # sup 標記（1: supervised, 0: robot）
                [ac.variance(o)],  # 初始 state 的 variance
                [],  # safety scores
            )

            if robosuite:
                simstates = [env.env.sim.get_state().flatten()]

            while step_count < obs_per_iter and not done:
                a_robot = ac.act(o)
                a_robot = np.clip(a_robot, -act_limit, act_limit)
                a_expert = expert_policy(o_robomimic)
                a_recovery = recovery_policy(o_robomimic)

                if not expert_mode:
                    # 只有在非 expert_mode 時才把 variance / safety 納入 estimates
                    estimates.append(ac.variance(o))
                    estimates2.append(ac.safety(o, a_robot))

                # --------------------------------------------------
                # expert_mode：由 human expert 控制
                # --------------------------------------------------
                if expert_mode:
                    a_expert = np.clip(a_expert, -act_limit, act_limit)

                    replay_buffer.store(o, a_expert)
                    online_burden += 1
                    # safety critic 對 expert action 的評分
                    risk.append(float(ac.safety(o, a_expert)))

                    if np.sum((a_robot - a_expert) ** 2) < switch2robot_thresh and (
                        not q_learning or ac.safety(o, a_robot) > switch2robot_thresh2
                    ):
                        print("Switch to Robot")
                        expert_mode = False
                        num_switch_to_robot += 1

                        o_robomimic, _, done, _ = env_robomimic.step(
                            a_expert
                        )  # FIXME: Should this be a_robot? @Sheng-Yu-Cheng
                        o2 = get_observation_for_thrifty_dagger(env)
                    else:
                        o_robomimic, _, done, _ = env_robomimic.step(a_expert)
                        o2 = get_observation_for_thrifty_dagger(env)

                    act.append(a_expert)
                    sup.append(1)  # 1 = supervised (human / recovery)

                    s_flag = env._check_success()
                    qbuffer.store(
                        o, a_expert, o2, int(s_flag), (ep_len + 1 >= horizon) or s_flag
                    )

                # --------------------------------------------------
                # safety_mode：由 recovery policy 控制
                # --------------------------------------------------
                elif safety_mode:
                    a_recovery = np.clip(a_recovery, -act_limit, act_limit)
                    replay_buffer.store(o, a_recovery)
                    risk.append(float(ac.safety(o, a_recovery)))

                    if np.sum((a_robot - a_recovery) ** 2) < switch2robot_thresh and (
                        not q_learning or ac.safety(o, a_robot) > switch2robot_thresh2
                    ):
                        print("Switch to Robot")
                        safety_mode = False
                        num_switch_to_robot += 1
                        o_robomimic, _, done, _ = env_robomimic.step(
                            a_recovery
                        )  # FIXME: Should this be a_robot? @Sheng-Yu-Cheng
                        o2 = get_observation_for_thrifty_dagger(env)
                    else:
                        o_robomimic, _, done, _ = env_robomimic.step(a_recovery)
                        o2 = get_observation_for_thrifty_dagger(env)

                    act.append(a_recovery)
                    sup.append(1)

                    s_flag = env._check_success()
                    qbuffer.store(
                        o,
                        a_expert,
                        o2,
                        int(s_flag),
                        (ep_len + 1 >= horizon) or s_flag,
                    )

                # --------------------------------------------------
                # 檢查是否需要切到 human：novelty / risk
                # --------------------------------------------------
                elif ac.variance(o) > switch2human_thresh:
                    print("Switch to Human (Novel)")
                    num_switch_to_human += 1
                    expert_mode = True
                    continue

                elif q_learning and ac.safety(o, a_robot) < switch2human_thresh2:
                    print("Switch to Human (Risk)")
                    num_switch_to_recovery += 1
                    safety_mode = True
                    continue

                # --------------------------------------------------
                # 一般情況：由 robot policy 控制
                # --------------------------------------------------
                else:
                    risk.append(float(ac.safety(o, a_robot)))
                    o_robomimic, _, done, _ = env_robomimic.step(a_robot)
                    o2 = get_observation_for_thrifty_dagger(env)

                    act.append(a_robot)
                    sup.append(0)

                    s_flag = env._check_success()
                    qbuffer.store(
                        o, a_robot, o2, int(s_flag), (ep_len + 1 >= horizon) or s_flag
                    )

                done = (ep_len + 1 >= horizon) or env._check_success()
                done_flags.append(done)
                rew.append(int(env._check_success()))

                o = o2
                obs.append(o)

                if robosuite:
                    simstates.append(env.env.sim.get_state().flatten())

                var.append(float(ac.variance(o)))

                step_count += 1
                ep_len += 1

            if done:
                ep_num += 1
            if ep_len >= horizon:
                fail_ct += 1

            total_env_interacts += ep_len

            episode_dict: Dict[str, Any] = {
                "obs": np.stack(obs),
                "act": np.stack(act),
                "done": np.array(done_flags),
                "rew": np.array(rew),
                "sup": np.array(sup),
                "var": np.array(var),
                "risk": np.array(risk),
                "beta_H": switch2human_thresh,
                "beta_R": switch2robot_thresh,
                "eps_H": switch2human_thresh2,
                "eps_R": switch2robot_thresh2,
                "simstates": np.array(simstates) if robosuite else None,
            }
            logging_data.append(episode_dict)

            pickle.dump(
                logging_data,
                open(
                    os.path.join(logger_kwargs["output_dir"], f"iter{epoch_idx}.pkl"),
                    "wb",
                ),
            )

            if robosuite:
                env.close()

            # online 更新 switching thresholds
            if len(estimates) > threshold_cfg.min_estimates_for_update:
                target_idx = int((1 - target_rate) * len(estimates))
                switch2human_thresh = sorted(estimates)[target_idx]
                switch2human_thresh2 = sorted(estimates2, reverse=True)[target_idx]
                switch2robot_thresh2 = sorted(estimates2)[int(0.5 * len(estimates))]

                print(
                    "len(estimates): {}, New switch thresholds: {} {} {}".format(
                        len(estimates),
                        switch2human_thresh,
                        switch2human_thresh2,
                        switch2robot_thresh2,
                    )
                )

        # --------------------------------------------------
        # 10-2. retrain policy / Q-risk
        # --------------------------------------------------
        ac_new, pi_optimizers, avg_loss_pi = retrain_policy(
            actor_critic,
            env,
            device,
            num_nets,
            ac_kwargs,
            replay_buffer,
            replay_size,
            obs_dim,
            act_dim,
            pi_lr,
            grad_steps,
            bc_epochs,
            epoch_idx,
            batch_size,
        )
        if ac_new is not None:
            ac = ac_new

        avg_loss_q = retrain_qrisk(
            ac,
            ac_targ,
            qbuffer,
            q_learning,
            num_test_episodes,
            expert_policy,
            recovery_policy,
            env,
            env_robomimic,
            act_limit,
            horizon,
            robosuite,
            logger_kwargs,
            pi_lr,
            bc_epochs,
            grad_steps,
            gamma,
            batch_size,
            qrisk_cfg,
            epoch_idx,
        )

        # --------------------------------------------------
        # 10-3. end-of-epoch logging
        # --------------------------------------------------
        logger.save_state(dict())
        log_epoch(
            logger,
            epoch_idx,
            ep_num,
            fail_ct,
            total_env_interacts,
            online_burden,
            num_switch_to_human,
            num_switch_to_recovery,
            num_switch_to_robot,
            avg_loss_pi,
            avg_loss_q,
            q_learning,
        )

        # --------------------------------------------------
        # 10-4. 早停條件：supervisor label 達上限
        # --------------------------------------------------
        if num_switch_to_human + num_switch_to_recovery >= max_expert_query:
            print("Reached max expert queries, stopping training.")
            break
