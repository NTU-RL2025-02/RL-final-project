from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import thrifty.algos.core as core
from thrifty.utils.logx import EpochLogger
import pickle
import os
import sys
import random

def get_observation_for_thrifty_dagger(env):
    obs_dict = env.env.observation_spec()
    return np.concat([obs_dict['robot0_proprio-state'], obs_dict['object-state']])

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        # 建立 obs / act 緩衝區，使用 numpy array 存資料
        # obs_dim, act_dim 可以是 scalar 或 tuple，透過 core.combined_shape 統一處理
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        # ptr: 目前寫入位置索引
        # size: 目前 buffer 中實際存了多少筆資料
        # max_size: buffer 最大容量
        self.ptr, self.size, self.max_size = 0, 0, size
        # device: 未來轉成 torch tensor 時要丟到哪個裝置 (cpu / cuda)
        self.device = device

    def store(self, obs, act):
        # 在目前 ptr 位置存入一組 (obs, act)，以 FIFO 方式覆蓋舊資料
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        # ptr 往後移動一格，超出就從 0 開始 (環狀 buffer)
        self.ptr = (self.ptr + 1) % self.max_size
        # 更新目前資料量 (最多到 max_size)
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        # 從現有資料 [0, self.size) 中隨機抽取 batch_size 筆索引
        idxs = np.random.randint(0, self.size, size=batch_size)
        # 根據 idxs 取出 obs 和 act
        batch = dict(obs=self.obs_buf[idxs], act=self.act_buf[idxs])
        # 轉成 torch tensor 並放到指定 device
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }

    def fill_buffer(self, obs, act):
        # 依序把一整批 (obs, act) 填進 buffer
        for i in range(len(obs)):
            self.store(obs[i], act[i])

    def save_buffer(self, name="replay"):
        # 把整個 buffer 狀態存成一個 pickle 檔，方便之後載入
        pickle.dump(
            {
                "obs_buf": self.obs_buf,
                "act_buf": self.act_buf,
                "ptr": self.ptr,
                "size": self.size,
            },
            open("{}_buffer.pkl".format(name), "wb"),
        )
        print("buf size", self.size)

    def load_buffer(self, name="replay"):
        # 從 pickle 檔中載入 buffer 狀態，覆寫目前的內容
        p = pickle.load(open("{}_buffer.pkl".format(name), "rb"))
        self.obs_buf = p["obs_buf"]
        self.act_buf = p["act_buf"]
        self.ptr = p["ptr"]
        self.size = p["size"]

    def clear(self):
        # 清空 buffer 的邏輯大小 (資料實際內容還在，但會被視為無效)
        self.ptr, self.size = 0, 0


class QReplayBuffer:
    # Replay buffer for training Qrisk
    def __init__(self, obs_dim, act_dim, size, device):
        # 用來存 Q-learning 需要的 transition (s, a, s', r, done)
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        # buffer metadata
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, next_obs, rew, done):
        # 存一筆 transition: (obs, act, next_obs, rew, done)
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        # done 轉成 float 存 (0.0 / 1.0)
        self.done_buf[self.ptr] = float(done)
        # 環狀 buffer 更新 ptr 與 size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, pos_fraction=None):
        # pos_fraction: 希望 batch 中有多少比例的樣本 reward 為 1
        # 這是為了加強對成功樣本的學習 (imbalanced data 時特別有用)
        if pos_fraction is not None:
            # 找出所有 reward != 0 的索引 (這裡視為正樣本)
            pos_size = min(
                len(tuple(np.argwhere(self.rew_buf).ravel())),
                int(batch_size * pos_fraction),
            )
            # 剩下的 batch 空間給負樣本
            neg_size = batch_size - pos_size
            # 從正樣本 index 中隨機抽 pos_size 個
            pos_idx = np.array(
                random.sample(tuple(np.argwhere(self.rew_buf).ravel()), pos_size)
            )
            # 從 reward=0 (且在現有 size 範圍內) 中抽 neg_size 個負樣本
            neg_idx = np.array(
                random.sample(
                    tuple(np.argwhere((1 - self.rew_buf)[: self.size]).ravel()),
                    neg_size,
                )
            )
            # 把正負樣本 index 合併
            idxs = np.hstack((pos_idx, neg_idx)).astype(np.int64)
            # 打亂順序
            np.random.shuffle(idxs)
        else:
            # 不特別控制正負比例，單純 uniform sampling
            idxs = np.random.randint(0, self.size, size=batch_size)
        # 根據 idxs 取出 batch 資料
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        # 轉成 torch tensor 放到對應 device
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }

    def fill_buffer(self, data):
        # 從一個包含 "obs", "act", "rew", "done" 的 rollouts 字典
        # 建立 Q-learning 用的 transition
        obs_dim = data["obs"].shape[1]
        act_dim = data["act"].shape[1]
        for i in range(len(data["obs"])):
            # 情況 1: done=1 且 rew=0，表示時間界線 (time limit)，不當作真正終止
            if data["done"][i] and not data["rew"][i]:  # time boundary, not really done
                continue
            # 情況 2: done=1 且 rew>0，表示成功結束 episode
            elif data["done"][i] and data["rew"][i]:  # successful termination
                # 這裡 next_obs 用 zero vector 表示 terminal
                self.store(
                    data["obs"][i],
                    data["act"][i],
                    np.zeros(obs_dim),
                    data["rew"][i],
                    data["done"][i],
                )
            else:
                # 一般 transition：下一步狀態為 obs[i+1]
                self.store(
                    data["obs"][i],
                    data["act"][i],
                    data["obs"][i + 1],
                    data["rew"][i],
                    data["done"][i],
                )

    def fill_buffer_from_BC(self, data, goals_only=False):
        """
        Load buffer from offline demos (only obs/act)
        goals_only: if True, only store the transitions with positive reward
        """
        # num_bc: BC 資料的長度
        num_bc = len(data["obs"])
        obs_dim = data["obs"].shape[1]
        # 這裡的邏輯是透過 action 的最後一維變化來偵測 episode 邊界
        # data["act"][i][-1] == 1 and data["act"][i+1][-1] == -1 表示新 episode 開始
        for i in range(num_bc - 1):
            if data["act"][i][-1] == 1 and data["act"][i + 1][-1] == -1:
                # 新 episode 開始時，將這個 transition 視為成功終止
                self.store(data["obs"][i], data["act"][i], np.zeros(obs_dim), 1, 1)
            elif not goals_only:
                # 否則如果不是 goals_only，就把中間 transition 存成 reward=0, done=0
                self.store(data["obs"][i], data["act"][i], data["obs"][i + 1], 0, 0)
        # 最後一個樣本當作 terminal，reward=1, done=1
        self.store(
            data["obs"][num_bc - 1], data["act"][num_bc - 1], np.zeros(obs_dim), 1, 1
        )

    def clear(self):
        # 清空邏輯大小 (資料還在，但不再被使用)
        self.ptr, self.size = 0, 0


def generate_offline_data(
    env,
    expert_policy,
    suboptimal_policy=None,
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
        if suboptimal_policy:
            suboptimal_policy.start_episode()
        # 重設環境
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


def thrifty(
    env,
    env_robomimic, 
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
    extra_obs_extractor=None,
    num_nets=5,
    target_rate=0.01,
    robosuite=False,
    robosuite_cfg=None,
    hg_dagger=None,
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
    hg_dagger: if not None, use this function as the switching condition (i.e. run HG-DAgger)
    q_learning: if True, train Q_risk safety critic
    gamma: discount factor for Q-learning
    num_test_episodes: run this many episodes after each iter without interventions
    init_model: initial NN weights
    """

    # 建立 logger，用來記錄實驗超參與訓練過程 (progress.txt 等)
    logger = EpochLogger(**logger_kwargs)
    _locals = locals()
    # config 中不存 env 這種無法 JSON 化的物件
    del _locals["env"]
    try:
        logger.save_config(_locals)
    except TypeError as e:
        # 某些不可序列化的物件會導致錯誤，這裡直接跳過
        print(f"[Warning] Could not save config as JSON: {e}")

    # device_idx >= 0 用對應的 GPU，否則使用 CPU
    if device_idx >= 0:
        device = torch.device("cuda", device_idx)
    else:
        device = torch.device("cpu")

    # 設定 random seed，確保實驗可重現
    torch.manual_seed(seed)
    np.random.seed(seed)
    if robosuite:
        # 若是 robosuite，將環境的 model.xml 存到 output_dir 方便之後 replay
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

    # initialize actor and classifier NN
    # 建立 Ensemble actor-critic (包含多個 policy 以及 Q-networks)
    ac = actor_critic(
        env.observation_space, env.action_space, device, num_nets=num_nets, **ac_kwargs
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
    # 建立 target network (用來做 Q-learning 的 target)
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        # target network 不需要做 gradient 更新
        p.requires_grad = False

    # Set up optimizers
    # 為 ensemble 中每一個 policy 建立一個 optimizer
    pi_optimizers = [Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)]
    # Q-network 使用同一個 optimizer，更新 q1, q2 兩個網路
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=pi_lr)

    # Set up model saving
    # 告訴 logger 之後要儲存哪一個 PyTorch 模型 (ac)
    logger.setup_pytorch_saver(ac)

    # Experience buffer
    # 建立一個 ReplayBuffer，存 BC 資料以及後續 online 收集到的資料
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
    )
    # 載入離線 expert rollouts (由 generate_offline_data 產生)
    input_data = pickle.load(open(input_file, "rb"))
    # shuffle and create small held out set to check valid loss
    num_bc = len(input_data["obs"])
    idxs = np.arange(num_bc)
    np.random.shuffle(idxs)
    # 前 90% 資料放進 replay buffer 當作 BC 訓練資料
    replay_buffer.fill_buffer(
        input_data["obs"][idxs][: int(0.9 * num_bc)],
        input_data["act"][idxs][: int(0.9 * num_bc)],
    )
    # 後 10% 當作 held-out set，用來估計 validation loss 與 threshold
    held_out_data = {
        "obs": input_data["obs"][idxs][int(0.9 * num_bc) :],
        "act": input_data["act"][idxs][int(0.9 * num_bc) :],
    }
    # Q-replay buffer，用來訓練 safety critic (Qrisk)
    qbuffer = QReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
    )
    # 從 BC 的離線資料初始化 Q buffer
    qbuffer.fill_buffer_from_BC(input_data)

    # Set up function for computing actor loss
    def compute_loss_pi(data, i):
        # data: batch 中包含 obs, act
        # i: 第幾個 ensemble 成員
        o, a = data["obs"], data["act"]
        # 使用第 i 個 policy 預測 action
        a_pred = ac.pis[i](o)
        # 使用 MSE loss (行為克隆)
        return torch.mean(torch.sum((a - a_pred) ** 2, dim=1))

    def compute_loss_q(data):
        # Q-loss: 使用 Bellman backup 進行 TD 誤差的 MSE
        o, a, o2, r, d = (
            data["obs"],
            data["act"],
            data["obs2"],
            data["rew"],
            data["done"],
        )
        # a2: 使用 ensemble 的平均 policy 對 next_obs 產生下一步 action
        with torch.no_grad():
            a2 = torch.mean(torch.stack([pi(o2) for pi in ac.pis]), dim=0)
        # 目前 Q1, Q2 的預測值
        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target network 上的 Q-values
            q1_t = ac_targ.q1(o2, a2)  # do target policy smoothing?
            q2_t = ac_targ.q2(o2, a2)
            # 取 min(Q1_t, Q2_t) 當作 target Q 值 (TD3 style)
            backup = r + gamma * (1 - d) * torch.min(q1_t, q2_t)
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        return loss_q1 + loss_q2

    def update_pi(data, i):
        # 對第 i 個 policy 做一次 gradient step
        pi_optimizers[i].zero_grad()
        loss_pi = compute_loss_pi(data, i)
        loss_pi.backward()
        pi_optimizers[i].step()
        return loss_pi.item()

    def update_q(data, timer):
        # 對 Q-network 做一次 gradient step
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        # 每隔幾步更新一次 target network
        if timer % 2 == 0:
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # 以 polyak averaging 更新 target 參數
                    p_targ.data.mul_(0.995)
                    p_targ.data.add_((1 - 0.995) * p.data)
        return loss_q.item()

    # Prepare for interaction with environment
    # online_burden: 對 human expert 請求 label 的總次數
    online_burden = 0  # how many labels we get from supervisor
    # num_switch_to_human: 因為 novelty (不確定性高) 而切換到 human 的次數
    num_switch_to_human = 0  # context switches (due to novelty)
    # num_switch_to_human2: 因為 risk (安全性低) 而切換到 human 的次數
    num_switch_to_human2 = 0  # context switches (due to risk)
    # num_switch_to_robot: 從 human / safety policy 切回 robot policy 的次數
    num_switch_to_robot = 0

    def test_agent(epoch=0):
        """Run test episodes"""
        # 用目前 policy ac 在環境中跑 num_test_episodes 回合，不會做 intervention
        obs, act, done, rew = [], [], [], []
        for j in range(num_test_episodes):
            ep_ret, ep_ret2, ep_len = 0, 0, 0
            o_robomimic, d = env_robomimic.reset(), False
            o = get_observation_for_thrifty_dagger(env)
            while not d:
                obs.append(o)
                # 用 ac.act 給出 action (ensemble 的決策)
                a = ac.act(o)
                a = np.clip(a, -act_limit, act_limit)
                act.append(a)
                o_robomimic, r, d, _ = env_robomimic.step(a)
                o = get_observation_for_thrifty_dagger(env)

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

    # 若 iters=0，表示只做 evaluation，不做訓練
    if iters == 0 and num_test_episodes > 0:  # only run evaluation.
        test_agent(0)
        sys.exit(0)

    ##############################################
    # ====== Pre-training with offline data ======
    ###############################################
    # 先用 BC 在 offline data 上訓練每個 policy (ensemble)
    for i in range(ac.num_nets):
        if ac.num_nets > 1:  # create new datasets via sampling with replacement
            # 對於 ensemble，使用 bootstrap resampling 形成不同 training set
            print("Net #{}".format(i))
            tmp_buffer = ReplayBuffer(
                obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
            )
            for _ in range(replay_buffer.size):
                idx = np.random.randint(replay_buffer.size)
                tmp_buffer.store(replay_buffer.obs_buf[idx], replay_buffer.act_buf[idx])
        else:
            # 單一 policy 就直接用 replay_buffer
            tmp_buffer = replay_buffer
        # 進行 bc_epochs 次 epoch，每個 epoch 做 grad_steps 次 update
        for j in range(bc_epochs):
            loss_pi = []
            for _ in range(grad_steps):
                batch = tmp_buffer.sample_batch(batch_size)
                loss_pi.append(update_pi(batch, i))
            # 驗證集上的 validation loss，用 held_out_data 計算
            validation = []
            for j in range(len(held_out_data["obs"])):
                a_pred = ac.act(held_out_data["obs"][j], i=i)
                a_sup = held_out_data["act"][j]
                validation.append(sum(a_pred - a_sup) ** 2)
            print("LossPi", sum(loss_pi) / len(loss_pi))
            print("LossValid", sum(validation) / len(validation))

    # estimate switch-back parameter and initial switch-to parameter from data
    # 使用 offline data 估計 switch threshold：
    #   - switch2robot_thresh: 判斷何時從 human/safety 切回 robot
    #   - switch2human_thresh: 判斷何時從 robot 切去 human (由 uncertainty 決定)
    discrepancies, estimates = [], []
    for i in range(replay_buffer.size):
        a_pred = ac.act(replay_buffer.obs_buf[i])
        a_sup = replay_buffer.act_buf[i]
        # discrepancies: robot 和 expert 的 action MSE
        discrepancies.append(sum((a_pred - a_sup) ** 2))
        # estimates: ac.variance 對該 obs 算出的不確定性
        estimates.append(ac.variance(replay_buffer.obs_buf[i]))
    heldout_discrepancies, heldout_estimates = [], []
    for i in range(len(held_out_data["obs"])):
        a_pred = ac.act(held_out_data["obs"][i])
        a_sup = held_out_data["act"][i]
        heldout_discrepancies.append(sum((a_pred - a_sup) ** 2))
        heldout_estimates.append(ac.variance(held_out_data["obs"][i]))
    # switch2robot_thresh: 使用整個 replay_buffer 的平均 discrepancy
    switch2robot_thresh = np.array(discrepancies).mean()
    # switch2human_thresh: 使用 held-out set 的 uncertainty 分位數
    target_idx = int((1 - target_rate) * len(heldout_estimates))
    switch2human_thresh = sorted(heldout_estimates)[target_idx]
    print("Estimated switch-back threshold: {}".format(switch2robot_thresh))
    print("Estimated switch-to threshold: {}".format(switch2human_thresh))
    # switch2human_thresh2, switch2robot_thresh2: 針對 risk-based (Qrisk) 的切換門檻
    switch2human_thresh2 = 0.48  # a priori guess: 48% discounted probability of success. Could also estimate from data
    switch2robot_thresh2 = 0.495
    # 清空 GPU cache (避免記憶體壓力)
    torch.cuda.empty_cache()
    # we only needed the held out set to check valid loss and compute thresholds, so we can get rid of it.
    replay_buffer.fill_buffer(held_out_data["obs"], held_out_data["act"])

    # 訓練過程中統計資訊
    total_env_interacts = 0  # 環境互動的總步數
    ep_num = 0  # 總 episode 數
    fail_ct = 0  # 失敗 episode 數 (超過 horizon)

    #######################################
    # ====== Main ThriftyDAgger Loop ======
    #######################################
    for t in range(iters + 1):
        # logging_data: 存這次 iter 的完整 rollouts 資料 (方便之後分析)
        logging_data = []  # for verbose logging
        estimates = []
        estimates2 = []  # refit every iter
        i = 0
        # t=0 時不做 data collection (只用 offline BC），留給 Q-training 使用
        if t == 0:  # skip data collection on iter 0 to train Q
            i = obs_per_iter
        # 每個 iter 收集 obs_per_iter 步的資料
        while i < obs_per_iter:
            # 每個 episode 開始前通知 expert / suboptimal policy
            expert_policy.start_episode()
            suboptimal_policy.start_episode()
            # expert_mode: 目前是否由 human expert 接管
            # safety_mode: 是否由 suboptimal/safety policy 接管
            o_robomimic, d = env_robomimic.reset(), False
            o = get_observation_for_thrifty_dagger(env)
            expert_mode, safety_mode, ep_len = False, False, 0
            # if robosuite:
            #     robosuite_cfg["INPUT_DEVICE"].start_control()
            # 儲存這個 episode 的軌跡
            obs, act, rew, done, sup, var, risk = (
                [o],
                [],
                [],
                [],
                [],
                [ac.variance(o)],
                [],
            )
            # robosuite 的話還會記錄 simstates 方便 replay
            if robosuite:
                simstates = [
                    env.env.sim.get_state().flatten()
                ]  # track this to replay trajectories after if desired.
            # 在這個 episode 中持續互動，直到 done 或收集步數達到 obs_per_iter
            while i < obs_per_iter and not d:
                # robot policy 的 action
                a = ac.act(o)
                a = np.clip(a, -act_limit, act_limit)
                a_expert = expert_policy(o_robomimic)
                a_suboptimal_policy = suboptimal_policy(o_robomimic)
                if not expert_mode:
                    # 非 expert_mode 時，把 variance 累積到 estimates，用來更新 switch 門檻
                    estimates.append(ac.variance(o))
                    # estimates2: safety score，用於 risk-based switching
                    estimates2.append(ac.safety(o, a))
                if expert_mode:
                    a_expert = expert_policy(o_robomimic)
                    a_expert = np.clip(a_expert, -act_limit, act_limit)
                    # 把 expert 提供的 (o, a_expert) 加進 replay_buffer，增加 BC data
                    replay_buffer.store(o, a_expert)
                    online_burden += 1
                    # risk: safety critic 對 expert action 的安全評估
                    risk.append(ac.safety(o, a_expert))
                    # 檢查是否要從 expert 切回 robot
                    if (hg_dagger and a_expert[3] != 0) or (
                        not hg_dagger
                        and sum((a - a_expert) ** 2) < switch2robot_thresh
                        and (not q_learning or ac.safety(o, a) > switch2robot_thresh2)
                    ):
                        print("Switch to Robot")
                        expert_mode = False
                        num_switch_to_robot += 1
                        o_robomimic, r, d, _ = env_robomimic.step(a_expert)
                        o2 = get_observation_for_thrifty_dagger(env)
                    else:
                        # 否則持續由 expert 控制
                        o_robomimic, r, d, _ = env_robomimic.step(a_expert)
                        o2 = get_observation_for_thrifty_dagger(env)
                    act.append(a_expert)
                    sup.append(1)  # sup=1 表示這一步是 supervised (human/suboptimal)
                    # 檢查是否成功
                    s = env._check_success()
                    # 把這筆 transition 放進 Q buffer 供 Q-learning 使用
                    qbuffer.store(o, a_expert, o2, int(s), (ep_len + 1 >= horizon) or s)
                elif safety_mode:
                    a_recovery = recovery_policy(o_robomimic)
                    a_recovery = np.clip(a_recovery, -act_limit, act_limit)
                    replay_buffer.store(o, a_recovery)
                    risk.append(float(ac.safety(o, a_recovery)))

                    if np.sum((a_robot - a_recovery) ** 2) < switch2robot_thresh and (
                        not q_learning or ac.safety(o, a_robot) > switch2robot_thresh2
                    ):
                        print("Switch to Robot")
                        safety_mode = False
                        num_switch_to_robot += 1
                        o_robomimic, r, d, _ = env_robomimic.step(a_expert)
                        o2 = get_observation_for_thrifty_dagger(env)
                    else:
                        # 持續由 safety policy 控制
                        o_robomimic, r, d, _ = env_robomimic.step(a_expert)
                        o2 = get_observation_for_thrifty_dagger(env)
                    act.append(a_expert)
                    sup.append(1)
                    s = env._check_success()
                    qbuffer.store(
                        o, a_expert, o2, int(s), (ep_len + 1 >= horizon) or s
                    )
                # hg-dagger switching for hg-dagger, or novelty switching for thriftydagger
                elif (hg_dagger and hg_dagger()) or (
                    not hg_dagger and ac.variance(o) > switch2human_thresh
                ):
                    # 若是 hg_dagger 模式，使用自訂條件
                    # 否則使用 uncertainty-based switching (variance 超過門檻)
                    print("Switch to Human (Novel)")
                    num_switch_to_human += 1
                    expert_mode = True
                    continue
                # second switch condition: if not novel, but also not safe
                elif (
                    not hg_dagger
                    and q_learning
                    and ac.safety(o, a) < switch2human_thresh2
                ):
                    # 若 model 看起來不 novel 但判斷不安全，切去 safety/human
                    print("Switch to Human (Risk)")
                    num_switch_to_human2 += 1
                    safety_mode = True
                    continue
                else:
                    # 一般情況：由 robot policy 控制
                    o_prev = o
                    risk.append(ac.safety(o, a))
                    o_robomimic, r, d, _ = env_robomimic.step(a)
                    o2 = get_observation_for_thrifty_dagger(env)
                    act.append(a)
                    sup.append(0)  # sup=0 表示由 robot policy 產生
                    s = env._check_success()
                    # 即使是 robot policy，也把 transition 存進 Q buffer
                    qbuffer.store(o_prev, a, o2, int(s), (ep_len + 1 >= horizon) or s)
                # robosuite 的 done 判定：episode 長度到 horizon 或成功
                d = (ep_len + 1 >= horizon) or env._check_success()
                done.append(d)
                # rew: 這裡 reward 簡化為是否成功 (0/1)
                rew.append(int(env._check_success()))
                # 前進到下一步狀態
                o = o2
                obs.append(o)
                if robosuite:
                    # 紀錄 simstate 以便之後可以重播整個 episode
                    simstates.append(env.env.sim.get_state().flatten())
                # var: 對新 state 的 variance
                var.append(ac.variance(o))
                i += 1
                ep_len += 1
            if d:
                # episode 結束
                ep_num += 1
            if ep_len >= horizon:
                # 若是因為 horizon 而終止，視為失敗
                fail_ct += 1
            total_env_interacts += ep_len
            # 把整個 episode 的 log 存到 logging_data
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
            # 將這個 iter 的 rollouts 存成 iter{t}.pkl
            pickle.dump(
                logging_data,
                open(logger_kwargs["output_dir"] + "/iter{}.pkl".format(t), "wb"),
            )
            if robosuite:
                env.close()
            # recompute thresholds from data after every episode
            # 收集到一定數量的 estimates 後，重算 switching threshold
            if len(estimates) > 25:
                target_idx = int((1 - target_rate) * len(estimates))
                switch2human_thresh = sorted(estimates)[target_idx]
                # estimates2 是 safety 的分數，越大越安全 (假設)
                switch2human_thresh2 = sorted(estimates2, reverse=True)[target_idx]
                # switch2robot_thresh2 大約取中位數，用來判斷是否切回 robot
                switch2robot_thresh2 = sorted(estimates2)[int(0.5 * len(estimates))]
                print(
                    "len(estimates): {}, New switch thresholds: {} {} {}".format(
                        len(estimates),
                        switch2human_thresh,
                        switch2human_thresh2,
                        switch2robot_thresh2,
                    )
                )

        if t > 0:
            # retrain policy from scratch
            # 每個 iter 結束後，使用目前累積的 replay_buffer 資料
            # 從頭重新訓練整個 actor_critic (ensemble)
            loss_pi = []
            ac = actor_critic(
                env.observation_space,
                env.action_space,
                device,
                num_nets=num_nets,
                **ac_kwargs,
            )
            pi_optimizers = [
                Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)
            ]
            for i in range(ac.num_nets):
                if ac.num_nets > 1:  # create new datasets via sampling with replacement
                    # bootstrap resampling，讓每個 ensemble 成員看到不同資料子集
                    tmp_buffer = ReplayBuffer(
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        size=replay_size,
                        device=device,
                    )
                    for _ in range(replay_buffer.size):
                        idx = np.random.randint(replay_buffer.size)
                        tmp_buffer.store(
                            replay_buffer.obs_buf[idx], replay_buffer.act_buf[idx]
                        )
                else:
                    tmp_buffer = replay_buffer
                # 訓練次數會隨著 iter 增加 (bc_epochs + t)
                for _ in range(grad_steps * (bc_epochs + t)):
                    batch = tmp_buffer.sample_batch(batch_size)
                    loss_pi.append(update_pi(batch, i))
        # retrain Qrisk
        if q_learning:
            # 若要訓練 Q-risk safety critic
            if num_test_episodes > 0:
                # 先跑 test_agent，使用目前的 policy 收集一批 rollouts 當作 Q-training 資料
                test_agent(t)  # collect samples offline from pi_R
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
                    # 每個 batch 固定一定比例的 positive samples (pos_fraction=0.1)
                    batch = qbuffer.sample_batch(batch_size // 2, pos_fraction=0.1)
                    loss_q.append(update_q(batch, timer=i))

        # end of epoch logging
        # 儲存當前模型權重
        logger.save_state(dict())
        print("Epoch", t)
        avg_loss_pi = 0.0
        avg_loss_q = 0.0
        if t > 0:
            avg_loss_pi = sum(loss_pi) / len(loss_pi)
            print("LossPi", sum(loss_pi) / len(loss_pi))
        if q_learning:
            avg_loss_q = sum(loss_q) / len(loss_q)
            print("LossQ", sum(loss_q) / len(loss_q))
        print("TotalEpisodes", ep_num)
        print("TotalSuccesses", ep_num - fail_ct)
        print("TotalEnvInteracts", total_env_interacts)
        print("OnlineBurden", online_burden)
        print("NumSwitchToNov", num_switch_to_human)
        print("NumSwitchToRisk", num_switch_to_human2)
        print("NumSwitchBack", num_switch_to_robot)

        # ===== Write to progress.txt using EpochLogger =====
        # success_rate: 成功 episode 所佔比例
        success_rate = (ep_num - fail_ct) / ep_num if ep_num > 0 else 0.0
        # 使用 logger.log_tabular 寫出 scalar 統計到 progress.txt
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
        # ============================================

        # 若已經超過最大允許 expert 查詢次數，提前停止訓練
        if online_burden >= max_expert_query:
            print("Reached max expert queries, stopping training.")
            break
