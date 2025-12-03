from gymnasium import Wrapper
class LunarLanderSuccessWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.success = False
        self.ep_reward = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        self.ep_reward += reward
        self.success = (self.ep_reward >= 200.0) and done
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.success = False
        self.ep_reward = 0.0
        return super().reset(**kwargs)

    def is_success(self):
        return self.success


class MazeWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.success = False

    def step(self, action):
        # TODO: define success condition
        return super().step(action)

    def reset(self, **kwargs):
        self.success = False
        return super().reset(**kwargs)

    def is_success(self):
        return self.success
