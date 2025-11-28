"""
thriftydagger.py: Thrifty DAgger algorithm implementation
"""

import os
import sys
import pickle
import itertools
from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam

import thrifty.algos.core as core
from thrifty.utils.logx import EpochLogger
from thrifty.algos.buffer import ReplayBuffer, QReplayBuffer


def generate_offline_data(
    env,
    expert_policy,
    recovery_policy=None,
    num_episodes=0,
    output_file="data.pkl",
    robosuite=False,
    robosuite_cfg=None,
    seed=0,
):
    # 透過 expert_policy 在 env 上 roll out num_episodes 回合
    # 收集 (obs, act) 並存成 offline BC 資料
    i, failures = 0, 0
    np.random.seed(seed)
    obs, act, rew = [], [], []
    # act_limit: action 空間的上界 (假設是對稱的)
    act_limit = env.action_space.high[0]
    while i < num_episodes:
        print("Episode #{}".format(i))
        # 通知 expert/suboptimal policy 開始新的 episode
        expert_policy.start_episode()
        if recovery_policy:
            recovery_policy.start_episode()  # NOTE: 為啥會有這？ by AaW
        o, total_ret, d, t = env.reset(), 0, False, 0
        curr_obs, curr_act = [], []
        # 如果是 robosuite，這裡原本有輸入設備控制 (註解掉)
        # if robosuite:
        #     robosuite_cfg["INPUT_DEVICE"].start_control()
        while not d:
            # 由 expert_policy 給出 action
            a = expert_policy(o)
            if a is None:
                # 若 expert 回傳 None，視為無法操作，直接終止
                d, r = True, 0
                continue
            # 將 action 限制在合法範圍內
            a = np.clip(a, -act_limit, act_limit)
            # 紀錄當下 obs 和 action
            curr_obs.append(o)
            curr_act.append(a)
            # 在環境中執行 action
            o, r, d, _ = env.step(a)
            if robosuite:
                # robosuite 的 done 判定與 reward 由 env._check_success() 控制
                d = (t >= robosuite_cfg["MAX_EP_LEN"]) or env._check_success()
                r = int(env._check_success())
            total_ret += r
            t += 1
        if robosuite:
            # robosuite 版本只保留成功 episode 的資料
            if total_ret > 0:  # only count successful episodes
                i += 1
                obs.extend(curr_obs)
                act.extend(curr_act)
            else:
                failures += 1
            env.close()
        else:
            # 一般環境不管成敗都增加 episode 計數
            i += 1
            obs.extend(curr_obs)
            act.extend(curr_act)
        print("Collected episode with return {} length {}".format(total_ret, t))
        rew.append(total_ret)
    # 印出回報統計與成功/失敗次數
    print("Ep Mean, Std Dev:", np.array(rew).mean(), np.array(rew).std())
    print("Num Successes {} Num Failures {}".format(num_episodes, failures))
    # 將 (obs, act) 存成 pickle 檔供之後 BC 使用
    pickle.dump({"obs": np.stack(obs), "act": np.stack(act)}, open(output_file, "wb"))


def compute_loss_pi(ac, data, i):
    o, a = data["obs"], data["act"]
    a_pred = ac.pis[i](o)
    return torch.mean(torch.sum((a - a_pred) ** 2, dim=1))


def compute_loss_q(ac, ac_targ, data, gamma):
    o, a, o2, r, d = (
        data["obs"],
        data["act"],
        data["obs2"],
        data["rew"],
        data["done"],
    )
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


def update_pi(ac, pi_optimizer, data, i):
    pi_optimizer.zero_grad()
    loss_pi = compute_loss_pi(ac, data, i)
    loss_pi.backward()
    pi_optimizer.step()
    return loss_pi.item()


def update_q(ac, ac_targ, q_optimizer, data, gamma, timer):
    q_optimizer.zero_grad()
    loss_q = compute_loss_q(ac, ac_targ, data, gamma)
    loss_q.backward()
    q_optimizer.step()

    if timer % 2 == 0:
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(0.995)
                p_targ.data.add_((1 - 0.995) * p.data)

    return loss_q.item()


def test_agent(
    expert_policy,
    recovery_policy,
    env,
    ac,
    num_test_episodes,
    act_limit,
    horizon,
    robosuite,
    logger_kwargs,
    epoch=0,
):
    """Run test episodes"""
    # 用目前 policy ac 在環境中跑 num_test_episodes 回合，不會做 intervention
    obs, act, done, rew = [], [], [], []
    for j in range(num_test_episodes):
        # 每個 episode 開始前，同樣通知 expert / suboptimal policy
        expert_policy.start_episode()
        recovery_policy.start_episode()
        o, d, ep_ret, ep_ret2, ep_len = env.reset(), False, 0, 0, 0
        while not d:
            obs.append(o)
            # 用 ac.act 給出 action (ensemble 的決策)
            a = ac.act(o)
            a = np.clip(a, -act_limit, act_limit)
            act.append(a)
            o, r, d, _ = env.step(a)
            if robosuite:
                # robosuite 的 success / done 檢查
                d = (ep_len + 1 >= horizon) or env._check_success()
                ep_ret2 += int(env._check_success())
                done.append(d)
                rew.append(int(env._check_success()))
            ep_ret += r
            ep_len += 1
        print("episode #{} success? {}".format(j, rew[-1]))
        if robosuite:
            env.close()
    # 計算成功率 (成功定義依照環境的 success flag)
    print("Test Success Rate:", sum(rew) / num_test_episodes)
    # 將測試 rollouts 存到 test-rollouts.pkl 給 Q-learning 使用
    pickle.dump(
        {
            "obs": np.stack(obs),
            "act": np.stack(act),
            "done": np.array(done),
            "rew": np.array(rew),
        },
        open("test-rollouts.pkl", "wb"),
    )
    # 同時也存一份到 output_dir，用 epoch 編號
    pickle.dump(
        {
            "obs": np.stack(obs),
            "act": np.stack(act),
            "done": np.array(done),
            "rew": np.array(rew),
        },
        open(logger_kwargs["output_dir"] + "/test{}.pkl".format(epoch), "wb"),
    )


def pretrain_policies(
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
):
    pi_optimizers = [Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)]

    for i in range(ac.num_nets):
        if ac.num_nets > 1:
            print("Net #{}".format(i))
            tmp_buffer = ReplayBuffer(
                obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
            )
            for _ in range(replay_buffer.size):
                idx = np.random.randint(replay_buffer.size)
                tmp_buffer.store(replay_buffer.obs_buf[idx], replay_buffer.act_buf[idx])
        else:
            tmp_buffer = replay_buffer

        for j in range(bc_epochs):
            loss_pi = []
            for _ in range(grad_steps):
                batch = tmp_buffer.sample_batch(batch_size)
                loss_pi.append(update_pi(ac, pi_optimizers[i], batch, i))

            validation = []
            for j in range(len(held_out_data["obs"])):
                a_pred = ac.act(held_out_data["obs"][j], i=i)
                a_sup = held_out_data["act"][j]
                validation.append(sum(a_pred - a_sup) ** 2)

            print("LossPi", sum(loss_pi) / len(loss_pi))
            print("LossValid", sum(validation) / len(validation))

    return pi_optimizers


def estimate_initial_thresholds(ac, replay_buffer, held_out_data, target_rate):
    """
    從 offline BC data 估計：
      - switch2robot_thresh: robot vs expert discrepancy 的平均
      - switch2human_thresh: ensemble variance 在 held-out set 上的 (1 - target_rate) 分位數
    """
    discrepancies = []
    for i in range(replay_buffer.size):
        a_pred = ac.act(replay_buffer.obs_buf[i])
        a_sup = replay_buffer.act_buf[i]
        # 用和原本一樣的 MSE 定義 discrepancy
        discrepancies.append(np.sum((a_pred - a_sup) ** 2))

    heldout_estimates = []
    for i in range(len(held_out_data["obs"])):
        heldout_estimates.append(ac.variance(held_out_data["obs"][i]))

    switch2robot_thresh = np.mean(discrepancies)

    # 和原本一樣：取 (1 - target_rate) 分位數
    target_idx = int((1 - target_rate) * len(heldout_estimates))
    switch2human_thresh = sorted(heldout_estimates)[target_idx]

    return switch2robot_thresh, switch2human_thresh


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
    recovery_policy=None,
    extra_obs_extractor=None,
    num_nets=5,
    target_rate=0.01,
    robosuite=False,
    robosuite_cfg=None,
    q_learning=False,
    gamma=0.9999,
    init_model=None,
    max_expert_query=2000,
):
    """
    obs_per_iter: environment steps per algorithm iteration
    num_nets: number of neural nets in the policy ensemble
    input_file: where initial BC data is stored (output of generate_offline_data())
    target_rate: desired rate of context switching
    robosuite: whether to enable robosuite specific code (and use robosuite_cfg)
    q_learning: if True, train Q_risk safety critic
    gamma: discount factor for Q-learning
    num_test_episodes: run this many episodes after each iter without interventions
    init_model: initial NN weights
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
        # 某些不可序列化的物件會導致錯誤，這裡直接跳過
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
        p.requires_grad = False  # target network 不需要梯度

    # ----------------------------------------------------------
    # 5. Q-network optimizer 與 model saving 設定
    # ----------------------------------------------------------
    # Q-network 使用同一個 optimizer，更新 q1, q2 兩個網路
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=pi_lr)

    # 告訴 logger 之後要儲存哪一個 PyTorch 模型 (ac)
    logger.setup_pytorch_saver(ac)

    # ----------------------------------------------------------
    # 6. 建立 replay buffers 並載入 offline data
    # ----------------------------------------------------------
    # 行為克隆用 replay buffer（obs, act）
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=replay_size,
        device=device,
    )

    # 載入離線 expert rollouts (由 generate_offline_data 產生的 data.pkl)
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
    # 9. 初始化統計量與 threshold
    # ----------------------------------------------------------
    # 計算 switch-back / switch-to thresholds
    switch2robot_thresh, switch2human_thresh = estimate_initial_thresholds(
        ac, replay_buffer, held_out_data, target_rate
    )
    print("Estimated switch-back threshold: {}".format(switch2robot_thresh))
    print("Estimated switch-to threshold: {}".format(switch2human_thresh))

    # 針對 risk-based (Qrisk) 的初始切換門檻
    switch2human_thresh2 = 0.48  # a priori guess: 48% discounted probability of success. Could also estimate from data
    switch2robot_thresh2 = 0.495

    # 清空 GPU cache（避免記憶體壓力）
    torch.cuda.empty_cache()

    # held-out set 之後不用了，把它也補進 replay buffer（讓之後 retrain 能看到全部資料）
    replay_buffer.fill_buffer(held_out_data["obs"], held_out_data["act"])

    # 訓練過程中統計資訊
    total_env_interacts = 0  # 環境互動的總步數
    ep_num = 0  # 總 episode 數
    fail_ct = 0  # 失敗 episode 數（超過 horizon）
    online_burden = 0  # supervisor 標註總數
    num_switch_to_human = 0  # 因 novelty 切到 human 次數
    num_switch_to_human2 = 0  # 因 risk 切到 human 次數
    num_switch_to_robot = 0  # 從 human/recovery 切回 robot 次數

    # ----------------------------------------------------------
    # 10. Main ThriftyDAgger Loop
    # ----------------------------------------------------------
    for t in range(iters + 1):
        # --------------------------------------------------
        # 1. 初始化本 iter 的暫存變數
        # --------------------------------------------------
        logging_data = []  # 存這次 iter 的完整 rollouts（方便離線分析）
        estimates = []  # 線上觀測到的 variance（用來更新 switch2human_thresh）
        estimates2 = []  # 線上觀測到的 safety score（用來更新 risk-based 門檻）
        i = 0

        # t = 0：只做 Q-training，不做資料收集
        if t == 0:
            i = obs_per_iter

        # --------------------------------------------------
        # 2. 收集 obs_per_iter 步的資料
        # --------------------------------------------------
        while i < obs_per_iter:
            # 每個 episode 開始前通知 expert / recovery policy
            expert_policy.start_episode()
            recovery_policy.start_episode()

            o, d = env.reset(), False
            expert_mode = False  # True：目前由 human expert 控制
            safety_mode = False  # True：目前由 recovery/safety policy 控制
            ep_len = 0

            # episode 累積的軌跡
            obs, act, rew, done, sup, var, risk = (
                [o],  # 初始 obs
                [],  # actions
                [],  # rewards（成功:1 / 失敗:0）
                [],  # done flags
                [],  # sup 標記（1: supervised, 0: robot）
                [ac.variance(o)],  # 初始 state 的 variance
                [],  # safety scores
            )

            # robosuite 額外記錄 sim states 方便 replay
            if robosuite:
                simstates = [env.env.sim.get_state().flatten()]

            # 單一 episode 中與環境互動，直到 done 或收集步數達到 obs_per_iter
            while i < obs_per_iter and not d:
                # robot policy 的 action（ensemble aggregation）
                a = ac.act(o)
                a = np.clip(a, -act_limit, act_limit)

                if not expert_mode:
                    # 只有在非 expert_mode 時才把 variance / safety 納入 estimates
                    estimates.append(ac.variance(o))
                    estimates2.append(ac.safety(o, a))

                # --------------------------------------------------
                # 2-1. expert_mode：由 human expert 控制
                # --------------------------------------------------
                if expert_mode:
                    a_expert = expert_policy(
                        extra_obs_extractor(env, env.env.env.latest_obs_dict)
                    )
                    a_expert = np.clip(a_expert, -act_limit, act_limit)

                    # 收集 expert demo 到 replay buffer（online BC）
                    replay_buffer.store(o, a_expert)
                    online_burden += 1

                    # safety critic 對 expert action 的評分
                    risk.append(ac.safety(o, a_expert))

                    # 判斷是否可以切回 robot policy
                    if sum((a - a_expert) ** 2) < switch2robot_thresh and (
                        not q_learning or ac.safety(o, a) > switch2robot_thresh2
                    ):
                        print("Switch to Robot")
                        expert_mode = False
                        num_switch_to_robot += 1
                        o2, _, d, _ = env.step(a_expert)
                    else:
                        # 否則持續由 expert 控制
                        o2, _, d, _ = env.step(a_expert)

                    act.append(a_expert)
                    sup.append(1)  # 1 = supervised (human / recovery)

                    s = env._check_success()
                    # 將 expert transition 也丟給 Q buffer（用於 Q-risk）
                    qbuffer.store(o, a_expert, o2, int(s), (ep_len + 1 >= horizon) or s)

                # --------------------------------------------------
                # 2-2. safety_mode：由 recovery / safety policy 控制
                # --------------------------------------------------
                elif safety_mode:
                    # NOTE: 目前用 recovery_policy 代替 suboptimal_policy
                    a_recovery_policy = recovery_policy(
                        extra_obs_extractor(env, env.env.env.latest_obs_dict)
                    )
                    a_recovery_policy = np.clip(
                        a_recovery_policy, -act_limit, act_limit
                    )

                    # 同樣把 safety policy 的 action 加進 replay buffer
                    replay_buffer.store(o, a_recovery_policy)
                    risk.append(ac.safety(o, a_recovery_policy))

                    # 判斷是否可以切回 robot policy
                    if sum((a - a_recovery_policy) ** 2) < switch2robot_thresh and (
                        not q_learning or ac.safety(o, a) > switch2robot_thresh2
                    ):
                        print("Switch to Robot")
                        safety_mode = False
                        num_switch_to_robot += 1
                        o2, _, d, _ = env.step(a_recovery_policy)
                    else:
                        o2, _, d, _ = env.step(a_recovery_policy)

                    act.append(a_recovery_policy)
                    sup.append(1)

                    s = env._check_success()
                    qbuffer.store(
                        o,
                        a_recovery_policy,
                        o2,
                        int(s),
                        (ep_len + 1 >= horizon) or s,
                    )

                # --------------------------------------------------
                # 2-3. 檢查是否需要「切到 human」：novelty / risk
                # --------------------------------------------------
                elif ac.variance(o) > switch2human_thresh:
                    # novel state：ensemble 對此 state 的不確定度太高
                    print("Switch to Human (Novel)")
                    num_switch_to_human += 1
                    expert_mode = True
                    # 不前進環境，下一輪 loop 由 expert_policy 來決定 action
                    continue

                elif q_learning and ac.safety(o, a) < switch2human_thresh2:
                    # risk 太高：safety critic 覺得 robot action 不安全
                    print("Switch to Human (Risk)")
                    num_switch_to_human2 += 1
                    safety_mode = True
                    # 同樣不前進環境
                    continue

                # --------------------------------------------------
                # 2-4. 一般情況：由 robot policy 控制
                # --------------------------------------------------
                else:
                    risk.append(ac.safety(o, a))
                    o2, _, d, _ = env.step(a)

                    act.append(a)
                    sup.append(0)  # 0 = robot 自己產生的動作

                    s = env._check_success()
                    # robot 的 transition 也餵進 Q buffer（on-policy data）
                    qbuffer.store(o, a, o2, int(s), (ep_len + 1 >= horizon) or s)

                # --------------------------------------------------
                # 2-5. 更新 episode / logging 狀態
                # --------------------------------------------------
                # robosuite 的終止條件：成功或 max horizon
                d = (ep_len + 1 >= horizon) or env._check_success()
                done.append(d)

                # reward：這裡簡化為 success flag（0/1）
                rew.append(int(env._check_success()))

                # 前進到下一個 state
                o = o2
                obs.append(o)

                if robosuite:
                    # 紀錄 simstate 以便之後可以重播整個 episode
                    simstates.append(env.env.sim.get_state().flatten())

                # 更新 variance 記錄
                var.append(ac.variance(o))

                i += 1
                ep_len += 1

            # -------- episode 結束後的統計與 logging_data 更新 --------
            if d:
                ep_num += 1
            if ep_len >= horizon:
                # 若是因為 horizon 而終止，視為 failure
                fail_ct += 1

            total_env_interacts += ep_len

            logging_data.append(
                {
                    "obs": np.stack(obs),
                    "act": np.stack(act),
                    "done": np.array(done),
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
            )

            # 每個 iter 持續覆寫 iter{t}.pkl，內容包含到目前為止所有 episodes
            pickle.dump(
                logging_data,
                open(logger_kwargs["output_dir"] + "/iter{}.pkl".format(t), "wb"),
            )

            if robosuite:
                env.close()

            # 當 estimates 累積到一定數量後，即時更新 switching thresholds
            if len(estimates) > 25:
                target_idx = int((1 - target_rate) * len(estimates))

                # variance 門檻：分位數
                switch2human_thresh = sorted(estimates)[target_idx]

                # safety 分數：越大越安全，故分位數要從大排到小
                switch2human_thresh2 = sorted(estimates2, reverse=True)[target_idx]

                # robot 門檻取中位數（heuristic）
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
        # 3. retrain policy from scratch（DAgger-aggregate BC）
        # --------------------------------------------------
        if t > 0:
            loss_pi = []

            # 重新初始化一個 actor_critic，並用目前累積的 replay_buffer 訓練
            ac = actor_critic(
                env.observation_space,
                env.action_space,
                device,
                num_nets=num_nets,
                **ac_kwargs,
            )

            # 更新 ensemble 每個成員的 optimizer
            pi_optimizers = [
                Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)
            ]

            for i in range(ac.num_nets):
                # bootstrap resampling：為每個 ensemble net 建立自己的 dataset
                if ac.num_nets > 1:
                    tmp_buffer = ReplayBuffer(
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        size=replay_size,
                        device=device,
                    )
                    for _ in range(replay_buffer.size):
                        idx = np.random.randint(replay_buffer.size)
                        tmp_buffer.store(
                            replay_buffer.obs_buf[idx],
                            replay_buffer.act_buf[idx],
                        )
                else:
                    tmp_buffer = replay_buffer

                # iter 越晚，訓練次數越多：grad_steps * (bc_epochs + t)
                for _ in range(grad_steps * (bc_epochs + t)):
                    batch = tmp_buffer.sample_batch(batch_size)
                    loss_pi.append(update_pi(ac, pi_optimizers[i], batch, i))

        # --------------------------------------------------
        # 4. retrain Q_risk（safety critic，選擇性）
        # --------------------------------------------------
        if q_learning:
            if num_test_episodes > 0:
                # 用目前 policy ac 跑幾個 episode，離線收集 Q-training data
                test_agent(
                    expert_policy,
                    recovery_policy,
                    env,
                    ac,
                    num_test_episodes,
                    act_limit,
                    horizon,
                    robosuite,
                    logger_kwargs,
                    epoch=t,
                )
                data = pickle.load(open("test-rollouts.pkl", "rb"))
                qbuffer.fill_buffer(data)
                os.remove("test-rollouts.pkl")

            # 重新設定 Q-network optimizer
            q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
            q_optimizer = Adam(q_params, lr=pi_lr)

            loss_q = []
            # 做 bc_epochs 個 epoch，每個 epoch 進行 grad_steps * 5 次 update
            for _ in range(bc_epochs):
                for i in range(grad_steps * 5):
                    batch = qbuffer.sample_batch(batch_size // 2, pos_fraction=0.1)
                    loss_q.append(
                        update_q(ac, ac_targ, q_optimizer, batch, gamma, timer=i)
                    )

        # --------------------------------------------------
        # 5. end-of-epoch logging（印出統計、寫 progress.txt）
        # --------------------------------------------------
        logger.save_state(dict())  # 儲存目前模型權重

        print("Epoch", t)
        avg_loss_pi = 0.0
        avg_loss_q = 0.0

        if t > 0:
            avg_loss_pi = sum(loss_pi) / len(loss_pi)
            print("LossPi", avg_loss_pi)

        if q_learning:
            avg_loss_q = sum(loss_q) / len(loss_q)
            print("LossQ", avg_loss_q)

        print("TotalEpisodes", ep_num)
        print("TotalSuccesses", ep_num - fail_ct)
        print("TotalEnvInteracts", total_env_interacts)
        print("OnlineBurden", online_burden)
        print("NumSwitchToNov", num_switch_to_human)
        print("NumSwitchToRisk", num_switch_to_human2)
        print("NumSwitchBack", num_switch_to_robot)

        # success_rate: 成功 episode 的比例
        success_rate = (ep_num - fail_ct) / ep_num if ep_num > 0 else 0.0

        logger.log_tabular("Epoch", t)
        logger.log_tabular("LossPi", avg_loss_pi)
        if q_learning:
            logger.log_tabular("LossQ", avg_loss_q)
        logger.log_tabular("TotalEpisodes", ep_num)
        logger.log_tabular("TotalSuccesses", ep_num - fail_ct)
        logger.log_tabular("SuccessRate", success_rate)
        logger.log_tabular("TotalEnvInteracts", total_env_interacts)
        logger.log_tabular("OnlineBurden", online_burden)
        logger.log_tabular("NumSwitchToNov", num_switch_to_human)
        logger.log_tabular("NumSwitchToRisk", num_switch_to_human2)
        logger.log_tabular("NumSwitchBack", num_switch_to_robot)

        logger.dump_tabular()

        # --------------------------------------------------
        # 6. 早停條件：supervisor label 達上限
        # --------------------------------------------------
        if online_burden >= max_expert_query:
            print("Reached max expert queries, stopping training.")
            break
