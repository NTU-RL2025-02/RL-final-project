import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from thrifty_gym.maze import FOUR_ROOMS_21x21
from gymnasium_robotics.envs.maze.maps import MEDIUM_MAZE


def load_rollouts(path: Path):
    """讀取 test-rollouts.pkl / test{epoch}.pkl。"""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expect dict in {path}, got {type(data)}")

    required_keys = {"obs"}
    if not required_keys.issubset(data.keys()):
        raise KeyError(f"pkl 檔缺少必要欄位: {required_keys - set(data.keys())}")

    return data


def split_episodes(data):
    """
    將攤平成一條時間序列的 rollouts，用 done 切回 episode。

    data:
        obs:  shape (T, obs_dim)
        act:  shape (T, act_dim) 或 (T,)
        done: shape (T,), bool
        rew:  shape (T,), int (0/1 success flag)

    回傳:
        List[np.ndarray]，其中每個元素是某一個 episode 的 obs，
        shape = (ep_len, obs_dim)
    """
    obs = np.asarray(data["obs"])  # (T, obs_dim)
    done = np.asarray(data["done"], bool)  # (T,)

    episodes_obs = []
    start = 0
    T = len(done)

    for t in range(T):
        if done[t]:
            ep_obs = obs[start : t + 1]  # [start, t] 含 t
            episodes_obs.append(ep_obs)
            start = t + 1

    # 若最後一段沒有 done=True，視情況要不要保留
    if start < T:
        ep_obs = obs[start:T]
        episodes_obs.append(ep_obs)
        print(f"[Warning] last segment without done=True, length={len(ep_obs)}")

    return episodes_obs


def main(
    input_path: Path, output_path: Path = None, maze_layout: str = "medium"
) -> None:
    """Load a trajectory pkl file and plot all trace."""
    # 1. 讀 pkl 並拆成 episodes
    data = load_rollouts(input_path)
    episodes_obs = split_episodes(data)

    print(f"Loaded {len(episodes_obs)} episodes from {input_path}")
    if episodes_obs:
        print("First episode obs shape:", episodes_obs[0].shape)

    # 2. 建立圖
    plt.figure(figsize=(8, 8))
    plt.title(f"Trajectories from {input_path.name}")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis("equal")

    # 3. 畫迷宮牆
    if maze_layout == "four_rooms":
        maze = FOUR_ROOMS_21x21
    elif maze_layout == "medium":
        maze = MEDIUM_MAZE
    else:
        raise ValueError("Unsupported maze layout: {}".format(maze_layout))

    width = len(maze[0])
    height = len(maze)
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell == 1:  # Wall
                x = j - width / 2
                y = -i + height / 2
                plt.fill_between([x, x + 1], [y - 1, y - 1], [y, y], color="yellow")

    # 4. 畫每條 episode 軌跡
    # 假設 obs 的前兩個維度是 (x, y)
    for ep_idx, ep_obs in enumerate(episodes_obs):
        if ep_obs.ndim != 2 or ep_obs.shape[1] < 2:
            raise ValueError(
                f"Episode {ep_idx} obs dim < 2，無法取 (x, y)，shape = {ep_obs.shape}"
            )

        xs = ep_obs[:, 0]
        ys = ep_obs[:, 1]

        # 畫軌跡線
        plt.plot(xs, ys, alpha=0.5, linewidth=1)

        # 起點/終點標記（可選）
        plt.scatter(xs[0], ys[0], marker="o", s=8)  # start
        plt.scatter(xs[-1], ys[-1], marker="x", s=8, linewidths=0.5)  # end

    # 5. 輸出或顯示
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a trajectory pkl file and plot all trace."
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to the trajectory pkl file.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save the plot image. If not provided, the plot will be shown instead.",
    )

    parser.add_argument(
        "--maze-layout",
        type=str,
        default="medium",
        help="Maze layout to use for plotting. Choices: 'medium', 'four_rooms'.",
    )

    args = parser.parse_args()
    main(args.input, args.output, args.maze_layout)
