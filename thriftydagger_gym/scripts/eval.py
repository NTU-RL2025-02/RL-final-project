"""
scripts/eval.py
Evaluation script for RecoveryDagger in maze environments.
Used to evaluate trained policies and visualize Q-value heatmaps.
Usage:
    python eval.py <path_to_trained_model.pt> [--environment MAZE_NAME] [--iters N] [--render] [--q_heatmap]
Options:
    --environment MAZE_NAME : Name of the maze environment to evaluate on. Choices: PointMaze_4rooms-v3, PointMaze_Complicated-v3, PointMaze_4rooms-v3-angle, PointMaze_4rooms-v3-angle-single-start
    --iters N               : Number of evaluation iterations (default: 100)
    --render                : Whether to render the environment during evaluation
    --q_heatmap             : Whether to draw Q-value heatmap
"""

from typing import Any
import numpy as np
import torch
import argparse
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
from thrifty_gym.utils.wrappers import (
    MazeWrapper,
    NoisyActionWrapper,
)

from thrifty_gym.maze import (
    FOUR_ROOMS_ANGLE,
    FOUR_ROOMS_ANGLE_SINGLE_START,
    FOUR_ROOMS_21_21,
    FOUR_ROOMS_21_21_LEFT_UP_RANDOM,
    COMPLICATED_MAZE,
)
from thrifty_gym.algos.rule_expert import RuleBasedExpert

import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation


@torch.no_grad()
def eval(
    env: Any,
    ac: Any,
    total_iterations: int,
    act_limit: float,
    horizon: int,
) -> float:

    if total_iterations <= 0:
        return 0.0

    total_success = 0

    for iter in trange(total_iterations):
        np.random.seed(iter)
        torch.manual_seed(iter)
        ball_traj = []
        episode_len = 0.0
        o, _ = env.reset()
        done = False

        while not done and episode_len < horizon:
            ball_x, ball_y = o[0], o[1]
            ball_traj.append([ball_x, ball_y])

            a = ac.act(o)
            a = np.clip(a, -act_limit, act_limit)

            o, _r, terminated, truncated, _info = env.step(a)

            success = env.is_success()
            total_success += success
            done = terminated or truncated or success

            episode_len += 1

        ball_x, ball_y = o[0], o[1]
        ball_traj.append([ball_x, ball_y])

    success_rate = total_success / total_iterations
    return success_rate, np.array(ball_traj)


def ij_to_xy(i, j, maze):
    width = len(maze[0])
    height = len(maze)
    x = j - width / 2
    y = height / 2 - i
    return x, y


def draw_q_heatmap(ac, maze, obs0):
    np.random.seed(0)
    torch.manual_seed(0)
    width = len(maze[0])
    height = len(maze)
    resolution = 20
    actions = [
        (-1, 1),
        (0, 1),
        (1, 1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ]
    q_table = np.zeros((height * resolution, width * resolution, len(actions)))

    for i in trange(height):
        for j in range(width):
            if maze[i][j] == 1 or maze[i][j] == "1":
                continue
            obs = obs0.copy()
            x, y = ij_to_xy(i, j, maze)
            for di in range(resolution):
                for dj in range(resolution):
                    ii = i * resolution + di
                    jj = j * resolution + dj
                    dx = di / resolution
                    dy = dj / resolution

                    obs[4] = x + dx
                    obs[5] = y + dy
                    obs[6] = 0
                    obs[7] = 0
                    for a_idx, a in enumerate(actions):
                        q_value = ac.safety(obs, a)
                        q_table[ii, jj, a_idx] = q_value

    # draw heatmap
    plt.figure(figsize=(10, 8))
    for a_idx, a in enumerate(actions):
        plt.subplot(3, 3, a_idx + 1)
        sns.heatmap(
            q_table[:, :, a_idx],
            cmap="viridis",
            cbar_kws={"label": "Q-value"},
            xticklabels=False,
            yticklabels=False,
            vmin=0.4,
        )
        plt.title(f"Action: {a}")
    plt.tight_layout()
    plt.show()


def make_env(args):
    gym.register_envs(gymnasium_robotics)

    env = None
    rander_mode = "human" if args.render else None
    if args.environment == "PointMaze_4rooms-v3":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode=rander_mode,
            maze_map=FOUR_ROOMS_21_21_LEFT_UP_RANDOM,
            max_episode_steps=1000,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, FOUR_ROOMS_21_21_LEFT_UP_RANDOM, touch_wall_distance=0.3)

    elif args.environment == "PointMaze_Complicated-v3":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode=rander_mode,
            maze_map=COMPLICATED_MAZE,
            max_episode_steps=1500,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, COMPLICATED_MAZE, touch_wall_distance=0.3)

    elif args.environment == "PointMaze_4rooms-v3-angle-single-start":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode=rander_mode,
            maze_map=FOUR_ROOMS_ANGLE_SINGLE_START,
            max_episode_steps=1100,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, FOUR_ROOMS_ANGLE_SINGLE_START)

    elif args.environment == "PointMaze_4rooms-v3-angle":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode=rander_mode,
            maze_map=FOUR_ROOMS_ANGLE,
            max_episode_steps=1100,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, FOUR_ROOMS_ANGLE)

    else:
        raise NotImplementedError("This environment is not implemented in this script.")

    return env


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path", type=str, help="Path to the .pt file.")
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of evaluation iterations."
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="PointMaze_4rooms-v3",
        help="Environment name. Choices: PointMaze_4rooms-v3, PointMaze_Complicated-v3, PointMaze_4rooms-v3-angle, PointMaze_4rooms-v3-angle-single-start",
    )
    parser.add_argument(
        "--noisy_scale",
        type=float,
        default=0,
        help="Scale of noise to add to actions when training the recovery policy. 0 means no noise.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether to render the environment during evaluation.",
    )
    parser.add_argument(
        "--q_heatmap", action="store_true", help="Whether to draw Q-value heatmap."
    )
    parser.set_defaults(render=False, q_heatmap=False)
    args = parser.parse_args()

    env = make_env(args)
    max_ep_len = 1000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open(args.pt_path, "rb") as f:
        ac = torch.load(f, weights_only=False, map_location=device)
        ac.device = device
        ac.eval()

    success_rate, ball_traj = eval(
        env,
        ac,
        total_iterations=args.iters,
        act_limit=env.action_space.high[0],
        horizon=max_ep_len,
    )
    print(f"Success rate over {args.iters} episodes: {success_rate}")
    if args.q_heatmap:
        draw_q_heatmap(ac, env.maze, env.reset()[0])
