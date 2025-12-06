"""
Train a Soft Actor-Critic (SAC) expert for Gymnasium's Point Maze environment.
The flow mirrors lunar_lander_sac.py: version prints, vectorized envs, eval and
checkpoint callbacks, best-model video recording, and performance plotting.
"""

import os
import platform
from importlib.metadata import version

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

import matplotlib.pyplot as plt
import tqdm


class CustomRewardFlattenObservation(FlattenObservation):
    """Flatten dict observations and do custom reward shaping."""

    def __init__(self, env, step_penalty: float = 0.01, room_bonus: float = 1.0):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.room_bonus = room_bonus
        self.prev_room = None  # 記住上一個房間

    def reset(self, **kwargs):
        # 先從原始 env 拿 dict obs
        obs_dict, info = self.env.reset(**kwargs)
        # 取出位置 (x, y) —— 來自 observation 的前兩維
        x, y = obs_dict["observation"][0], obs_dict["observation"][1]
        self.prev_room = self._get_room_id(x, y)
        # 再把 obs flatten 給 SB3 用
        flat_obs = self.observation(obs_dict)
        return flat_obs, info

    def step(self, action):
        # 注意：這裡直接呼叫 self.env.step，拿 dict obs
        obs_dict, reward, terminated, truncated, info = self.env.step(action)

        # 取目前位置
        x, y = obs_dict["observation"][0], obs_dict["observation"][1]
        current_room = self._get_room_id(x, y)
        

        shaped_reward = reward

        # 1. 每一個時間步給小 penalty
        shaped_reward -= self.step_penalty

        # 2. 如果跨房間，就加一個 bonus
        if (
            self.prev_room is not None
            and current_room is not None
            and current_room != self.prev_room
        ):
            shaped_reward += self.room_bonus

        # 更新 prev_room
        self.prev_room = current_room

        # 最後 flatten obs 回傳給 SAC
        flat_obs = self.observation(obs_dict)
        return flat_obs, shaped_reward, terminated, truncated, info

    def _get_room_id(self, x: float, y: float):
        """
        粗略用座標象限區分四個房間：
        - y >= 0, x <= 0 → 左上
        - y >= 0, x > 0  → 右上
        - y < 0,  x <= 0 → 左下
        - y < 0,  x > 0  → 右下
        之後你如果知道實際牆的位置，可以再把 0 換成更精準的 threshold。
        """
        if y >= 0 and x <= 0:
            return 0  # room 0
        elif y >= 0 and x > 0:
            return 1  # room 1
        elif y < 0 and x <= 0:
            return 2  # room 2
        else:
            return 3  # room 3


def safe_version(pkg: str) -> str:
    """Return package version or a placeholder if missing."""
    try:
        return version(pkg)
    except Exception:
        return "not-installed"


print(f"Python Version: {platform.python_version()}")
print(f"Torch Version: {safe_version('torch')}")
print(f"Is Cuda Available: {torch.cuda.is_available()}")
print(f"Cuda Version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"Gymnasium Version: {safe_version('gymnasium')}")
print(f"Numpy Version: {safe_version('numpy')}")
print(f"Stable Baselines3 Version: {safe_version('stable_baselines3')}", flush = True)

RL_TYPE = "SAC"
ENV_ID = "PointMaze_UMaze-v3"  # choose any PointMaze variant you prefer
LOG_DIR = os.path.join("./logs", ENV_ID)
NAME_PREFIX = "point_maze_sac"
MAZE_FILE = "maze_4room.txt"
STEP_PENALTY = 0.01  # small penalty per step to encourage faster solutions
ROOM_BONUS = 1.0    # bonus for entering a new room

with open(MAZE_FILE) as file:
    MAZE = [list(map(lambda x: int(x) if x in ["0", "1"] else x, line.split())) for line in file.readlines()]
        
# Training/evaluation kwargs keep rendering off for speed; video env enables RGB frames.
TRAIN_EVAL_ENV_KWARGS = {"render_mode": None, 'maze_map': MAZE}
VIDEO_ENV_KWARGS = {"render_mode": "rgb_array", "maze_map": MAZE}

# Vector env settings
N_ENVS = 16
EVAL_ENVS = 1

# Evaluation callback frequency (timesteps) and evaluation episodes
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 10

VIDEO_RECORD_FREQ = 50_000

# Total training steps; Point Maze is sparse so budget generously for expert-quality policy.
TOTAL_TIMESTEPS = 1_200_000

os.makedirs(LOG_DIR, exist_ok=True)


class PeriodicVideoRecorder(BaseCallback):
    """Record a rollout every `record_freq` training timesteps."""

    def __init__(self, record_freq: int, video_length: int = 1_000):
        super().__init__()
        self.record_freq = record_freq
        self.video_length = video_length
        self._last_recorded_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_recorded_step < self.record_freq:
            return True

        name_prefix = f"{NAME_PREFIX}_step{self.num_timesteps}"
        record_video(
            self.model,
            name_prefix=name_prefix,
            video_length=self.video_length,
        )
        self._last_recorded_step = self.num_timesteps
        return True


def main() -> None:
    # Inspect the observation/action spaces after flattening the dict observation.
    sample_env = CustomRewardFlattenObservation(
        gym.make(ENV_ID, **TRAIN_EVAL_ENV_KWARGS),
        step_penalty=STEP_PENALTY,
        room_bonus=ROOM_BONUS,
    )
    print("Observation Space:", sample_env.observation_space)
    print("Action Space:", sample_env.action_space, flush=True)
    sample_env.close()

    # Training and evaluation environments (vectorized).
    env = make_vec_env(
        ENV_ID,
        n_envs=N_ENVS,
        seed=0,
        wrapper_class=CustomRewardFlattenObservation,
        wrapper_kwargs={"step_penalty": STEP_PENALTY, "room_bonus": ROOM_BONUS},
        env_kwargs=TRAIN_EVAL_ENV_KWARGS,
    )
    env_val = make_vec_env(
        ENV_ID,
        n_envs=EVAL_ENVS,
        seed=1,
        wrapper_class=CustomRewardFlattenObservation,
        wrapper_kwargs={"step_penalty": STEP_PENALTY, "room_bonus": ROOM_BONUS},
        env_kwargs=TRAIN_EVAL_ENV_KWARGS,
    )

    # Callbacks: save checkpoints and evaluate/track the best model.
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        render=False,
        deterministic=True,
        n_eval_episodes=N_EVAL_EPISODES,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=EVAL_FREQ,
        save_path=os.path.join(LOG_DIR, "checkpoint"),
    )

    video_callback = PeriodicVideoRecorder(record_freq=VIDEO_RECORD_FREQ)

    callback_list = CallbackList([checkpoint_callback, eval_callback, video_callback])

    # Initialize SAC
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
    )

    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=callback_list,
    )

    # Save final model
    model.save(os.path.join(LOG_DIR, "final_model"))

    # Evaluate the latest model
    mean_reward, std_reward = evaluate_policy(
        model, env_val, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
    )
    print(f"Final model mean reward: {mean_reward:.2f} +/- {std_reward:.2f}", flush=True)

    env.close()
    env_val.close()

    # Load and evaluate the best model saved by EvalCallback
    env_val = make_vec_env(
        ENV_ID,
        n_envs=EVAL_ENVS,
        seed=2,
        wrapper_class=CustomRewardFlattenObservation,
        wrapper_kwargs={"step_penalty": STEP_PENALTY, "room_bonus": ROOM_BONUS},
        env_kwargs=TRAIN_EVAL_ENV_KWARGS,
    )
    best_model_path = os.path.join(LOG_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        best_model = SAC.load(best_model_path, env=env_val)
        mean_reward, std_reward = evaluate_policy(
            best_model, env_val, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
        )
        print(f"Best model mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        record_video(best_model)
    else:
        print("best_model.zip not found yet; skipping video.")

    env_val.close()
    plot_learning_curve()


def record_video(
    model: SAC,
    name_prefix: str = NAME_PREFIX,
    video_length: int = 1_000,
    env_kwargs=None,
    step_penalty: float = STEP_PENALTY,
    room_bonus: float = ROOM_BONUS,
) -> None:
    """Record a short rollout of the model to LOG_DIR."""
    env_kwargs = env_kwargs or VIDEO_ENV_KWARGS
    env = make_vec_env(
        ENV_ID,
        n_envs=1,
        seed=42,
        wrapper_class=CustomRewardFlattenObservation,
        wrapper_kwargs={"step_penalty": step_penalty, "room_bonus": room_bonus},
        env_kwargs=env_kwargs,
    )
    env = VecVideoRecorder(
        env,
        LOG_DIR,
        video_length=video_length,
        record_video_trigger=lambda x: x == 0,
        name_prefix=name_prefix,
    )

    obs = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        env.render()
        if dones.any():
            obs = env.reset()

    env.close()


def plot_learning_curve() -> None:
    """Plot evaluation rewards saved by EvalCallback (evaluations.npz)."""
    eval_path = os.path.join(LOG_DIR, "evaluations.npz")
    if not os.path.exists(eval_path):
        print("evaluations.npz not found; skipping plot.")
        return

    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]

    mean_results = np.mean(results, axis=1)
    std_results = np.std(results, axis=1)

    plt.figure()
    plt.plot(timesteps, mean_results, label="mean reward")
    plt.fill_between(
        timesteps, mean_results - std_results, mean_results + std_results, alpha=0.3
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title(f"{RL_TYPE} on {ENV_ID}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
