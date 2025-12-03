import numpy as np
import torch
import torch.nn as nn
from core import MLPQFunction

class Recovery():
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
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
            hidden_sizes=(256, 256),
            activation=nn.ReLU,
            num_nets=5,
            alpha=0.9, 
            eta=0.5
        ):
        """
        Parameters:
        q_networks: list of callables, each taking (obs, act) and returning scalar Q(s,a)
            這部分我還沒implement好
        q_risk: for accumuate_risk
        alpha: R_t = alpha * R_{t-1} + (1-alpha) * indicator(用到eta當threshold)
        """
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.num_nets = num_nets
        self.q_networks = [
            MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
            for _ in range(num_nets)
        ]
        self.alpha = alpha
        self.q_risk = q_risk
        self.eta = eta
        self.device = device
        self.R_t = 0.0  # initial risk score for AccumulateRisk
    
    def five_q_networks(self, obs, act):
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
        return mean_q, var_q
    
    def gradient_ascent(self, obs, init_action, steps=20, lr=0.01, action_bounds=(-1.0, 1.0)):
        """
        Gradient ascent to find action that maximizes Q(s,a).
        還沒有用5個Q network的平均值來做gradient ascent
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
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        a = torch.as_tensor(init_action, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        best_q = float('-inf')
        best_a = a.clone().detach()
        
        for _ in range(steps):
            # 計算平均 Q 值
            q_vals = []
            for q_net in self.q_networks:
                q_vals.append(q_net(o, a))
            mean_q = torch.mean(torch.stack(q_vals))
            
            # Backward pass
            if a.grad is not None:
                a.grad.zero_()
            mean_q.backward()
            
            # Gradient ascent step
            with torch.no_grad():
                a.data = a.data + lr * a.grad
                # Clip action to bounds
                a.data = torch.clamp(a.data, action_bounds[0], action_bounds[1])
            
            # Track best action
            with torch.no_grad():
                current_q = mean_q.item()
                if current_q > best_q:
                    best_q = current_q
                    best_a = a.clone().detach()
        
        return best_a.cpu().numpy().squeeze()

    
    def choose_action(self, obs, candidate_acts):
        # a_star = self.gradient_ascent(obs, init_action, )
        raise NotImplementedError
    
    def accumulate_risk(self, obs, act):
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
    

    """
    記得寫如何呼叫
    要問：
    gradient_ascent -> action_bounds: action 的範圍，用來clip結果，我不太確定要不要這個action_bounds但每個AI都這樣建議我
    """

    