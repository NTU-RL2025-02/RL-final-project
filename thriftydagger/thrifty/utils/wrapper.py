import gymnasium
import numpy as np


class ObsCachingWrapper:
    """
    Lightweight wrapper that caches raw robosuite dict observations.
    Not a Gymnasium wrapper because the base robosuite env is not a Gym Env.
    """

    def __init__(self, env):
        self.env = env
        self.latest_obs_dict = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.latest_obs_dict = obs
        return obs

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        self.latest_obs_dict = obs
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class CustomWrapper(gymnasium.Env):
    def __init__(self, env, render):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gripper_closed = False
        self.viewer = env.viewer
        self.robots = env.robots
        self._render = render

    def _step(self, action):
        """
        Normalize step outputs to (obs, reward, done, info) even if the base env
        follows the Gymnasium API and returns terminated/truncated separately.
        """
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        return result

    def reset(self):
        res = self.env.reset()  # o ?O obs ?V?q (23,)
        o = res[0] if isinstance(res, tuple) else res
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = -1
        for _ in range(10):
            o, r, d, info = self._step(settle_action)
            # print(o, r, d, info)  # ?u???n debug ?A?L
            self.render()
        self.gripper_closed = False
        return o

    def step(self, action):
        # abstract 10 actions as 1 action
        # get rid of x/y rotation, which is unintuitive for remote teleop
        action_ = action.copy()
        action_[3] = 0.0
        action_[4] = 0.0
        self._step(action_)
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = action[-1]
        for _ in range(10):
            r1, r2, r3, r4 = self._step(settle_action)
            self.render()
        if action[-1] > 0:
            self.gripper_closed = True
        else:
            self.gripper_closed = False
        return r1, r2, r3, r4

    def _check_success(self):
        return self.env._check_success()

    def render(self):
        if self._render:
            self.env.render()
