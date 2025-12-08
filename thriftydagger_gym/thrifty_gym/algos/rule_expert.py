import numpy as np


class RuleBasedExpert:
    def __init__(self, maze, maze_reward):
        self.maze = maze
        self.maze_reward = maze_reward
        self.maze_height = len(maze)
        self.maze_width = len(maze[0])

    def start_episode(self):
        pass

    def xy_to_cell_rowcol(self, x, y):

        i = int(-y + self.maze_height / 2)
        j = int(x + self.maze_width / 2)

        i = np.clip(i, 0, self.maze_height - 1)
        j = np.clip(j, 0, self.maze_width - 1)

        return i, j

    def __call__(self, observation) -> np.ndarray:
        """
        根據 observation 回傳一個 rule-based 的 action。

        observation: (x, y, vx, vy)
        """
        x, y = observation[0], observation[1]
        v_x, v_y = observation[2], observation[3]
        i, j = self.xy_to_cell_rowcol(x, y)
        direction = self.maze_reward[i][j]

        if direction == "U":
            base = np.array([0.0, 1.0], dtype=np.float32)
            damp = np.array([-v_x, 0.0], dtype=np.float32)
        elif direction == "D":
            base = np.array([0.0, -1.0], dtype=np.float32)
            damp = np.array([-v_x, 0.0], dtype=np.float32)
        elif direction == "R":
            base = np.array([1.0, 0.0], dtype=np.float32)
            damp = np.array([0.0, -v_y], dtype=np.float32)
        elif direction == "L":
            base = np.array([-1.0, 0.0], dtype=np.float32)
            damp = np.array([0.0, -v_y], dtype=np.float32)
        else:
            base = np.zeros(2, dtype=np.float32)
            damp = np.array([-v_x, -v_y], dtype=np.float32)
        return np.clip(base + damp, -1.0, 1.0)
