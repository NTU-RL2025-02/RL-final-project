import pickle
import random
import numpy as np
import torch
import thrifty.algos.core as core


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
