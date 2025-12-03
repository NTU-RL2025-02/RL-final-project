import numpy as np
import torch

class Recovery():
    """
    Recovery helper class for:
      - maintaining an ensemble of Q-networks
      - choosing best action via argmax Q
      - accumulating a smoothed risk score R_t

    q_networks: list of callables, each taking (obs, act) and returning scalar Q(s,a)
    risk_q:     callable for Q_risk(s,a), can be one of the q_networks or a separate net
    """
    def __init__(self, q_networks, q_risk, alpha=0.9, eta=0.5):
        """
        Parameters:
        q_networks: list of callables, each taking (obs, act) and returning scalar Q(s,a)
            這部分我還沒implement好
        q_risk: for accumuate_risk
        alpha: R_t = alpha * R_{t-1} + (1-alpha) * indicator(用到eta當threshold)
        """
        self.q_networks = q_networks
        self.alpha = alpha
        self.q_risk = q_risk
        self.eta = eta
        self.R_t = 0.0  # initial risk score for AccumulateRisk
    
    def five_q_networks(self, obs, act):
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        a = torch.as_tensor(act, dtype=torch.float32).unsqueeze(0)
        q_values = []
        for q_net in self.q_networks:
            with torch.no_grad():
                q_val = q_net(o, a).item()  # get scalar value
            q_values.append(q_val)
        q_values = np.asarray(q_values, dtype=np.float32)
        mean_q = float(np.mean(q_values))
        var_q = float(np.var(q_values))
        return mean_q, var_q
    
    def gradient_ascent(self, obs):
        raise NotImplementedError
    
    def choose_action(self, obs, candidate_acts):
        raise NotImplementedError
    
    def accumulate_risk(self, obs, act):
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        a = torch.as_tensor(act, dtype=torch.float32).unsqueeze(0)
        q_risk_value = self.q_risk(o, a).item()
        indicator = 1.0 if q_risk_value < self.eta else 0.0
        self.R_t = self.alpha * self.R_t + (1 - self.alpha) * indicator
        return self.R_t
    

    """
    記得寫如何呼叫
    """

    """
    原本判斷要不要接Q是這樣寫：
    elif q_learning and ac.safety(o, a_robot) < switch2human_thresh2:
        print("Switch to Human (Risk)")
        num_switch_to_human2 += 1
        safety_mode = True
        continue
    """