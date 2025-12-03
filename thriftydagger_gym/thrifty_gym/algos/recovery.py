import numpy as np
import torch
import torch.nn as nn
from core import MLPQFunction


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
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_nets = num_nets
        self.alpha = 0.9  # for accumulate_risk
        self.eta = 0.5    # threshold for risk indicator
        self.q_networks = [
            MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)
            for _ in range(num_nets)
        ]
        self.q_risk = q_risk
        self.variance_weight = variance_weight
        self.R_t = 0.0  # initial risk score for AccumulateRisk

    def objective(self, obs, action):
        raise NotImplementedError

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

    def choose_action(self, obs, init_action):
        # FIXME: 我的初步想法？應該是這樣，但不太確定是不是這樣子就好
        return self.gradient_ascent(obs, init_action)


class QRecovery(Recovery):
    def __init__(self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__(action_space)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.q_networks = [
            MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)
        ]
    def objective(self, obs, action):
        # 直接最大化Q value
        return self.q_networks[0](obs, action).view(-1)[0]


class FiveQRecovery(Recovery):
    def __init__( 
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        num_nets=5,
        variance_weight=1.0,
    ):
        super().__init__(num_nets=num_nets, variance_weight=variance_weight)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.variance_weight = variance_weight
        self.q_networks = [
            MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)
            for _ in range(num_nets)
        ]

    def five_q_networks(self, obs, act):
        """
        評估給定的 observation 和 action 在5個Q networks上的表現
        
        Returns:
        --------
        mean_q: 5個Q values的平均值
        var_q: 5個Q values的variance
        q_values: 5個Q values的list
        """
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        a = torch.as_tensor(act, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = []
        for q_net in self.q_networks:
            with torch.no_grad():
                q_val = q_net(o, a).item()  # get scalar value
                q_values.append(q_val)
        q_values = np.asarray(q_values, dtype=np.float32)
        mean_q = float(np.mean(q_values))
        var_q = float(np.var(q_values))
        return mean_q, var_q, q_values
    
    def objective(self, obs, action):
        """
        f(a) = mean_Q(a) - lambda * var_Q(a)
        這裡才用到 5 個 Q + variance。
        """
        q_vals = [q(obs, action).view(-1)[0] for q in self.q_networks]
        q_stack = torch.stack(q_vals)  # (num_q,)
        mean_q = q_stack.mean()
        var_q = q_stack.var(unbiased=False)
        return mean_q - self.variance_weight * var_q
    


    """
    使用5個Q networks來計算mean和variance，並通過gradient ascent優化組合目標函數：
    f(a) = mean_Q(a) - lambda * var_Q(a)
    我的想法是，這是一個multi-objective optimization問題，所以直接把mean_Q和var_Q組合成一個函數
    """


