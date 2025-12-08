"""
Plot trajectories stored in pkl (rule-based offline data) or hdf5.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MAZE_FILE = BASE_DIR / "maze_4room.txt"


def load_pkl_rollouts(path: Path):
    """讀取 test-rollouts.pkl / test{epoch}.pkl。"""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expect dict in {path}, got {type(data)}")

    required_keys = {"obs", "done"}
    if not required_keys.issubset(data.keys()):
        raise KeyError(f"pkl 檔缺少必要欄位: {required_keys - set(data.keys())}")

    return data


def split_pkl_episodes(data):
    """
    將攤平成一條時間序列的 rollouts，用 done 或 episode_starts 切回 episode。
    """
    obs = np.asarray(data["obs"])  # (T, obs_dim)
    done = np.asarray(data["done"], bool)  # (T,)
    starts = np.asarray(data.get("episode_starts", []), bool)

    episodes_obs: List[np.ndarray] = []
    T = len(obs)

    if starts.size == T and starts.any():
        start_indices = list(np.nonzero(starts)[0]) + [T]
        for a, b in zip(start_indices[:-1], start_indices[1:]):
            episodes_obs.append(obs[a:b])
    else:
        start = 0
        for t in range(T):
            if done[t]:
                ep_obs = obs[start : t + 1]  # [start, t] 含 t
                episodes_obs.append(ep_obs)
                start = t + 1
        if start < T:
            ep_obs = obs[start:T]
            episodes_obs.append(ep_obs)
            print(f"[Warning] last segment without done=True, length={len(ep_obs)}")

    return np.array(episodes_obs, dtype=object)


def load_hdf5_trajectories(
    path: Path, hdf5_training_traj: bool = False, hdf5_testing_traj: bool = False
):
    """讀取 trajectories.hdf5"""
    episode_obs, policy_using = [], []
    with h5py.File(path, "r") as f:
        if hdf5_training_traj:
            for ep in f["training"]:
                episode_obs.append(f[f"training/{ep}/position"][()])
                policy_using.append(f[f"training/{ep}/policy"][()])
        if hdf5_testing_traj:
            for ep in f["testing"]:
                episode_obs.append(f[f"testing/{ep}/position"][()])
                policy_using.append(None)

    return episode_obs, policy_using


def load_maze_layout(layout: str, maze_file: Path | None):
    """
    Load maze layout from a txt file or fallback to simple placeholders.
    """
    if maze_file is not None and maze_file.exists():
        with maze_file.open() as f:
            return [line.split() for line in f.readlines()]

    if layout == "four_rooms":
        # Minimal four-rooms outline if file missing.
        return [
            ["1"] * 5,
            ["1", "r", "r", "r", "1"],
            ["1", "r", "1", "r", "1"],
            ["1", "r", "r", "r", "1"],
            ["1"] * 5,
        ]
    # Default: medium maze placeholder
    return [
        ["1", "1", "1", "1", "1"],
        ["1", "r", "r", "r", "1"],
        ["1", "r", "r", "r", "1"],
        ["1", "r", "r", "r", "1"],
        ["1", "1", "1", "1", "1"],
    ]


def is_wall(cell: Union[int, str]) -> bool:
    if isinstance(cell, (int, float)):
        return int(cell) == 1
    if isinstance(cell, str):
        return cell == "1"
    return False


def main(
    input_type: str,
    input_path: Path,
    output_path: Path = None,
    maze_layout: str = "medium",
    sample_amount: int | None = None,
    hdf5_training_traj: bool = False,
    hdf5_testing_traj: bool = False,
    maze_file: Path | None = None,
) -> None:
    """Load a trajectory pkl file and plot all trace."""
    # 1. 讀 pkl 或 hdf5 並拆成 episodes
    if input_type == "pkl":
        data = load_pkl_rollouts(input_path)
        episodes_obs = split_pkl_episodes(data)
        policy_using = [None] * len(episodes_obs)
    elif input_type == "hdf5":
        episodes_obs, policy_using = load_hdf5_trajectories(
            input_path, hdf5_training_traj, hdf5_testing_traj
        )

    print(f"Loaded {len(episodes_obs)} episodes from {input_path}")
    if len(episodes_obs) != 0:
        print("First episode obs shape:", episodes_obs[0].shape)

    if sample_amount is not None and len(episodes_obs) > 0:
        sample_amount = min(sample_amount, len(episodes_obs))
        idx = np.random.choice(len(episodes_obs), size=sample_amount, replace=False)
        episodes_obs = [episodes_obs[i] for i in idx]
        print(f"Sampled {len(episodes_obs)} episodes from all trajectories")

    # 2. 建立圖
    plt.figure(figsize=(8, 8))
    plt.title(f"Trajectories from {input_path.name}")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis("equal")

    # 3. 畫迷宮牆
    maze = load_maze_layout(maze_layout, maze_file)
    width = len(maze[0])
    height = len(maze)
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if is_wall(cell):  # Wall
                x = j - width / 2
                y = -i + height / 2
                plt.fill_between([x, x + 1], [y - 1, y - 1], [y, y], color="yellow")

    # 4. 畫每條 episode 軌跡
    # 假設 obs 的前兩個維度是 (x, y)
    for ep_idx, ep_obs in enumerate(episodes_obs):
        ep_obs = np.asarray(ep_obs)
        if ep_obs.ndim != 2 or ep_obs.shape[1] < 2:
            raise ValueError(
                f"Episode {ep_idx} obs dim < 2，無法取 (x, y)，shape = {ep_obs.shape}"
            )

        xs = ep_obs[:, 0]
        ys = ep_obs[:, 1]

        if input_type == "pkl":
            plt.plot(xs, ys, alpha=0.5, linewidth=1)
            plt.scatter(xs[0], ys[0], marker="o", s=8)  # start
            plt.scatter(xs[-1], ys[-1], marker="x", s=8, linewidths=0.5)  # end
        elif input_type == "hdf5":
            if policy_using[ep_idx] is not None:
                plt.plot(xs, ys, alpha=0.5, linewidth=1)
                plt.scatter(xs[0], ys[0], marker="o", s=8)  # start
                plt.scatter(xs[-1], ys[-1], marker="x", s=8, linewidths=0.5)  # end
            else:
                previous_policy = policy_using[ep_idx][0]
                x_seg, y_seg = [xs[0]], [ys[0]]
                for x, y, p in zip(xs[1:], ys[1:], policy_using[ep_idx]):
                    x_seg.append(x)
                    y_seg.append(y)
                    if p != previous_policy:
                        plt.plot(
                            x_seg,
                            y_seg,
                            alpha=0.5,
                            linewidth=1,
                            color=("black", "red", "green")[previous_policy],
                            label=("robot", "expert", "recovery")[previous_policy],
                        )
                        previous_policy = p
                        x_seg, y_seg = [x], [y]
                plt.plot(
                    x_seg,
                    y_seg,
                    alpha=0.5,
                    linewidth=1,
                    color=("black", "red", "green")[previous_policy],
                    label=("robot", "expert", "recovery")[previous_policy],
                )
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

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--pkl_input",
        type=Path,
        help="Path to the trajectory pkl file.",
    )

    input_group.add_argument(
        "--hdf5_input",
        type=Path,
        help="Path to the trajectory hdf5 file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None ,
        help="Path to save the plot image. If not provided, the plot will be shown instead.",
    )

    parser.add_argument(
        "--maze-layout",
        type=str,
        default="medium",
        help="Maze layout to use for plotting. Choices: 'medium', 'four_rooms'.",
    )
    parser.add_argument(
        "--maze-file",
        type=Path,
        default=DEFAULT_MAZE_FILE,
        help="Optional custom maze txt file (overrides maze-layout).",
    )

    parser.add_argument(
        "--sample-amount",
        type=int,
        default=None,
        help="trajectory sample amount from the dataset",
    )

    # argument for hdf5 trajectory input
    parser.set_defaults(hdf5_training_traj=False)
    parser.add_argument(
        "--hdf5-training-traj",
        action="store_true",
        dest="hdf5_training_traj",
        help="If using hdf5 file as input, then enable training trajectory observation.",
    )

    parser.set_defaults(hdf5_testing_traj=False)
    parser.add_argument(
        "--hdf5-testing-traj",
        action="store_true",
        dest="hdf5_testing_traj",
        help="If using hdf5 file as input, then enable training trajectory observation.",
    )

    args = parser.parse_args()
    main(
        "pkl" if args.pkl_input is not None else "hdf5",
        args.pkl_input if args.pkl_input is not None else args.hdf5_input,
        args.output,
        args.maze_layout,
        args.sample_amount,
        args.hdf5_training_traj,
        args.hdf5_testing_traj,
        args.maze_file,
    )
