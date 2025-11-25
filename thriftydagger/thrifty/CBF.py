import numpy as np

try:
    import cvxpy as cp
except ImportError:
    cp = None

class CBFController:
    """
    CBF-based QP filter for Robosuite NutAssembly.
    Simplified Version: Only protects against Table Collision (Z-axis).
    """

    def __init__(self, table_z_limit=0.86, action_scale=0.05, alpha=10.0):
        """
        :param table_z_limit: float, 桌面高度限制 (default: 0.86)
        :param action_scale: float, OSC controller 的 output_max (default: 0.05)
        :param alpha: float, CBF 的鬆弛係數 (gamma), 越大越激進
        """
        self.z_limit = table_z_limit
        self.action_scale = action_scale
        self.alpha = alpha
        
        self.cvxpy_available = cp is not None
        if not self.cvxpy_available:
            print("[Warning] cvxpy not available; CBFController will passthrough actions.")

    def get_safe_action(self, obs, nominal_action):
        """
        :param obs: Robosuite 的 observation 字典 (必須包含 'robot0_eef_pos')
        :param nominal_action: 原始 Policy 輸出的動作 (shape: 7)
        :return: 安全的動作 (shape: 7)
        """
        # 0. 基本數據處理
        nominal_action = np.asarray(nominal_action, dtype=np.float32)
        
        # 檢查 cvxpy 是否可用，以及 obs 是否包含必要資訊
        if not self.cvxpy_available or 'robot0_eef_pos' not in obs:
            return np.clip(nominal_action, -1.0, 1.0)

        # 1. 獲取狀態
        # robot0_eef_pos shape=(3,) -> [x, y, z]
        current_eef_pos = obs['robot0_eef_pos']
        
        # 2. 分離動作
        # 我們只優化前 3 維 (x, y, z)，後 4 維 (orientation + gripper) 直接保留
        u_nom_pos = nominal_action[:3]
        u_rest = nominal_action[3:]
        
        # 3. 建立優化問題 (只針對 Position Action u_opt)
        u_opt = cp.Variable(3)

        constraints = []
        
        # --- 限制 A: Action Space Bounds [-1, 1] ---
        constraints += [u_opt >= -1.0, u_opt <= 1.0]

        # --- 限制 B: Table Collision (Z-axis only) ---
        # Barrier Function h(x) = z_current - z_limit >= 0
        # h_dot = v_z = action_scale * u_z
        # Constraint: h_dot + alpha * h >= 0
        #           => action_scale * u_z + alpha * (z_current - z_limit) >= 0
        
        h_val = current_eef_pos[2] - self.z_limit
        
        constraints.append(
            self.action_scale * u_opt[2] + self.alpha * h_val >= 0
        )

        # 4. 目標函數: 最小化與原始動作的差距 (Least Squares)
        objective = cp.Minimize(cp.sum_squares(u_opt - u_nom_pos))
        
        # 5. 求解
        prob = cp.Problem(objective, constraints)
        
        try:
            # 嘗試求解，OSQP 對這類問題最快
            prob.solve(solver=cp.OSQP, warm_start=True)
        except Exception as e:
            print(f"[Warning] CBF QP solve error: {e}")
            # 求解器報錯時，為了安全，位置動作歸零 (煞車)，保留姿態動作
            return np.concatenate([np.zeros(3), u_rest])

        # 6. 結果處理
        if u_opt.value is None:
            # 無解 (通常代表已經穿透或嚴重違反限制)，執行煞車
            return np.concatenate([np.zeros(3), u_rest])
            
        # 組合回 7 維動作
        safe_pos_action = np.array(u_opt.value).squeeze()
        full_safe_action = np.concatenate([safe_pos_action, u_rest])
        
        return np.clip(full_safe_action, -1.0, 1.0)