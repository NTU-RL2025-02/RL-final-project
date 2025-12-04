import numpy as np
import torch
import torch.nn as nn
from thrifty_gym.algos.core import MLPQFunction


class Recovery:
    """
    Recovery helper class for:
      - maintaining an ensemble of Q-networks
      - choosing best action via argmax Q
      - accumulating a smoothed risk score R_t

    q_networks: list of callables, each taking (obs, act) and returning scalar Q(s,a)
    risk_q:     callable for Q_risk(s,a), can be one of the q_networks or a separate net
    """

    def __init__(
        self,
        q_risk,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        num_nets=5,
        variance_weight=1.0,
    ):
        """
        Parameters:
        q_networks: list of callables, each taking (obs, act) and returning scalar Q(s,a)
            這部分我還沒implement好
        q_risk: for accumuate_risk
        alpha: R_t = alpha * R_{t-1} + (1-alpha) * indicator(用到eta當threshold)
        variance_weight: lambda 參數，用來平衡 mean_Q 和 var_Q
                        較大的值表示更重視減小 variance
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = 0.9  # for accumulate_risk
        self.eta = 0.5    # threshold for risk indicator
        self.R_t = 0.0    # initial risk score for AccumulateRisk

        self.q_risk = q_risk
        self.variance_weight = variance_weight
        self.q_networks = []

        if observation_space is not None and action_space is not None and num_nets > 0:
            obs_dim = observation_space.shape[0]
            act_dim = action_space.shape[0]
            self.q_networks = [
                MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)
                for _ in range(num_nets)
            ]
    
    def objective(self, obs, action):
        raise NotImplementedError

    def start_episode(self):
        """讓 thrifty() 可以在每個 episode 開頭 reset 狀態。"""
        self.R_t = 0.0

    def gradient_ascent(
        self, obs, init_action, steps=20, lr=0.01, action_bounds=(-1.0, 1.0)
    ):
        """
        最大化 self.objective(obs, act)
        
        Parameters:
        -----------
        obs: observation
        init_action: 原本我們的IL agent選的action，作為初始化
        steps: gradient ascent 迭代次數
        lr: learning rate
        action_bounds: action 的範圍，用來clip結果

        Returns:
        --------
        best_action: numpy array of optimized action
        """
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.as_tensor(init_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = a.clone().detach().requires_grad_(True)

        best_obj = float("-inf")
        best_a = a.clone().detach()

        for _ in range(steps):
            objective = self.objective(o, a)

            # Backward pass
            if a.grad is not None:
                a.grad.zero_()
            objective.backward()

            # Gradient ascent step
            with torch.no_grad():
                a.data = a.data + lr * a.grad
                # Clip action to bounds
                a.data = torch.clamp(a.data, action_bounds[0], action_bounds[1])

            if objective.item() > best_obj:
                best_obj = objective.item()
                best_a = a.clone().detach()

        return best_a.cpu().numpy().squeeze()

    # 以下的函式並不是一個recovery policy，只是另外一種判斷是否要切到expert的方式，以後要再更改回原程式碼中
    def accumulate_risk(self, obs, act, alpha=0.9, eta=0.5):
        """
        原本判斷要不要接Q是這樣寫：
        elif q_learning and ac.safety(o, a_robot) < switch2human_thresh2:
            print("Switch to Human (Risk)")
            num_switch_to_human2 += 1
            safety_mode = True
            continue
        現在判斷要用accumulate_risk
        """
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        a = torch.as_tensor(act, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_risk_value = self.q_risk(o, a).item()
        indicator = 1.0 if q_risk_value < self.eta else 0.0
        self.R_t = self.alpha * self.R_t + (1 - self.alpha) * indicator
        return self.R_t


class QRecovery(Recovery):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        q_risk=None,
    ):
        super().__init__(
            q_risk=q_risk,
            observation_space=observation_space,
            action_space=action_space,
            hidden_sizes=hidden_sizes,
            activation=activation,
            num_nets=1,         # 單一 Q-network
            variance_weight=1.0,
        )

    def objective(self, obs, action):
        # 直接最大化 Q value
        return self.q_networks[0](obs, action).view(-1)[0]
    
    def run(self, obs, init_action, steps=20, lr=0.01, action_bounds=(-1.0, 1.0)):
        # return an action
        return self.gradient_ascent(obs, init_action, steps, lr, action_bounds)


class FiveQRecovery(Recovery):
    def __init__( 
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        num_nets=5,
        variance_weight=1.0,
        q_risk=None,
    ):
        super().__init__(
            q_risk=q_risk,
            observation_space=observation_space,
            action_space=action_space,
            hidden_sizes=hidden_sizes,
            activation=activation,
            num_nets=num_nets,
            variance_weight=variance_weight,
        )

    def objective(self, obs, action):
        """
        f(a) = mean_Q(a) - lambda * var_Q(a)
        這裡用到5個Q functions
        """
        q_vals = [q(obs, action).view(-1)[0] for q in self.q_networks]
        q_stack = torch.stack(q_vals)  # (num_q,)
        mean_q = q_stack.mean()
        var_q = q_stack.var(unbiased=False)
        return mean_q - self.variance_weight * var_q
    
    def run(self, obs, init_action, steps=20, lr=0.01, action_bounds=(-1.0, 1.0)):
        # return an action
        return self.gradient_ascent(obs, init_action, steps, lr, action_bounds)

    """
    使用5個Q networks來計算mean和variance，並通過gradient ascent優化組合目標函數：
    f(a) = mean_Q(a) - lambda * var_Q(a)
    我的想法是，這是一個multi-objective optimization問題，所以直接把mean_Q和var_Q組合成一個函數
    """


