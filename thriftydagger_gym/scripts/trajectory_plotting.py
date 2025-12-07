import pickle
import h5py
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from thrifty_gym.maze import FOUR_ROOMS_21x21, FOUR_ROOMS_ANGLE
from gymnasium_robotics.envs.maze.maps import MEDIUM_MAZE


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


def load_hdf5_trajectories(
    path: Path, hdf5_training_traj: bool = False, hdf5_testing_traj: bool = False
):
    """讀取 trajectories.hdf5"""
    episode_obs, policy_using = [], []
    with h5py.File(path, "r") as f:
        if hdf5_training_traj:
            for traj in f["training"]:
                episode_obs.append(traj["position"])
                policy_using.append(traj["policy"])
        if hdf5_testing_traj:
            for traj in f["testing"]:
                episode_obs.append(traj["position"])
                policy_using.append(None)
    return np.array(episode_obs), policy_using


def main(
    input_type: str,
    input_path: Path,
    output_path: Path = None,
    maze_layout: str = "medium",
    hdf5_training_traj: bool = False,
    hdf5_testing_traj: bool = False,
) -> None:
    """Load a trajectory pkl file and plot all trace."""
    # 1. 讀 pkl 或 hdf5 並拆成 episodes
    if input_type == "pkl":
        data = load_pkl_rollouts(input_path)
        episodes_obs = split_pkl_episodes(data)
        policy_using = [None] * episodes_obs.shape[0]
    elif input_type == "hdf5":
        episodes_obs, policy_using = load_hdf5_trajectories(
            input_path, hdf5_training_traj, hdf5_testing_traj
        )

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
        maze = FOUR_ROOMS_ANGLE
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

        if input_type == "pkl":
            # 畫軌跡線
            plt.plot(xs, ys, alpha=0.5, linewidth=1)

            # 起點/終點標記（可選）
            plt.scatter(xs[0], ys[0], marker="o", s=8)  # start
            plt.scatter(xs[-1], ys[-1], marker="x", s=8, linewidths=0.5)  # end
        elif input_type == "hdf5":
            if policy_using[ep_idx] == None:
                # testing stage
                # 畫軌跡線
                plt.plot(xs, ys, alpha=0.5, linewidth=1)

                # 起點/終點標記（可選）
                plt.scatter(xs[0], ys[0], marker="o", s=8)  # start
                plt.scatter(xs[-1], ys[-1], marker="x", s=8, linewidths=0.5)  # end
            else:
                # training stage
                previous_policy = policy_using[0]
                x_seg, y_seg = [xs[0]], [ys[0]]
                for i, x, y, p in enumerate(zip(xs[1:], ys[1:], policy_using)):
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
                plt.legend()
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
        default=None,
        help="Path to save the plot image. If not provided, the plot will be shown instead.",
    )

    parser.add_argument(
        "--maze-layout",
        type=str,
        default="medium",
        help="Maze layout to use for plotting. Choices: 'medium', 'four_rooms'.",
    )

    # argument for hdf5 trajectory input
    parser.add_argument(
        "--hdf5_training_traj",
        action="store_true",
        dest="render",
        help="If using hdf5 file as input, then enable training trajectory observation.",
    )
    parser.set_defaults(hdf5_training_traj=False)

    parser.add_argument(
        "--hdf5_testing_traj",
        action="store_true",
        dest="render",
        help="If using hdf5 file as input, then enable training trajectory observation.",
    )
    parser.set_defaults(hdf5_testing_traj=False)

    args = parser.parse_args()
    main(
        "pkl" if args.pkl_input is not None else "hdf5",
        args.pkl_input if args.pkl_input is not None else args.hdf5_input,
        args.output,
        args.maze_layout,
        args.hdf5_training_traj,
        args.hdf5_testing_traj,
    )
