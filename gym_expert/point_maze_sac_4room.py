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
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

import matplotlib.pyplot as plt
import tqdm


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
print(f"Stable Baselines3 Version: {safe_version('stable_baselines3')}")

RL_TYPE = "SAC"
ENV_ID = "PointMaze_UMaze-v3"  # choose any PointMaze variant you prefer
LOG_DIR = os.path.join("./logs", ENV_ID)
NAME_PREFIX = "point_maze_sac"
MAZE_FILE = "maze_4room.txt"

with open(MAZE_FILE) as file:
    MAZE = [list(map(int, line.split())) for line in file.readlines()]
        
# Training/evaluation kwargs keep rendering off for speed; video env enables RGB frames.
TRAIN_EVAL_ENV_KWARGS = {"render_mode": None, 'maze_map': MAZE}
VIDEO_ENV_KWARGS = {"render_mode": "rgb_array"}

# Vector env settings
N_ENVS = 4
EVAL_ENVS = 1

# Evaluation callback frequency (timesteps) and evaluation episodes
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 10

# Total training steps; Point Maze is sparse so budget generously for expert-quality policy.
TOTAL_TIMESTEPS = 1_200_000

os.makedirs(LOG_DIR, exist_ok=True)


def main() -> None:
    # Inspect the observation/action spaces after flattening the dict observation.
    sample_env = FlattenObservation(gym.make(ENV_ID, **TRAIN_EVAL_ENV_KWARGS))
    print("Observation Space:", sample_env.observation_space)
    print("Action Space:", sample_env.action_space)
    sample_env.close()

    # Training and evaluation environments (vectorized).
    env = make_vec_env(
        ENV_ID,
        n_envs=N_ENVS,
        seed=0,
        wrapper_class=FlattenObservation,
        env_kwargs=TRAIN_EVAL_ENV_KWARGS,
    )
    env_val = make_vec_env(
        ENV_ID,
        n_envs=EVAL_ENVS,
        seed=1,
        wrapper_class=FlattenObservation,
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

    callback_list = CallbackList([checkpoint_callback, eval_callback])

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
    print(f"Final model mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    env_val.close()

    # Load and evaluate the best model saved by EvalCallback
    env_val = make_vec_env(
        ENV_ID,
        n_envs=EVAL_ENVS,
        seed=2,
        wrapper_class=FlattenObservation,
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


def record_video(model: SAC) -> None:
    """Record a short rollout of the best model to LOG_DIR."""
    env = make_vec_env(
        ENV_ID,
        n_envs=1,
        seed=42,
        wrapper_class=FlattenObservation,
        env_kwargs=VIDEO_ENV_KWARGS,
    )
    env = VecVideoRecorder(
        env,
        LOG_DIR,
        video_length=1_000,
        record_video_trigger=lambda x: x == 0,
        name_prefix=NAME_PREFIX,
    )

    obs = env.reset()
    for _ in range(1_000):
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
