import numpy as np

try:
    import cvxpy as cp
except Exception:
    cp = None


class CBFController:
    """
    Simple CBF-based QP filter to nudge nominal actions toward safety.
    Uses placeholder dynamics (Lf=0, Lg=1) for a kinematic model.
    Customize h, Lf_h, and Lg_h for your system.
    """

    def __init__(self, act_limit, alpha=1.0):
        self.act_limit = act_limit
        self.alpha = alpha
        self.cvxpy_available = cp is not None
        if not self.cvxpy_available:
            print("[Warning] cvxpy not available; CBFController will passthrough actions.")

    def h(self, obs):
        # Placeholder barrier: stay within unit radius from origin on the first 3 dims.
        pos = np.asarray(obs)[0:3]
        return 1.0 - np.linalg.norm(pos)

    def Lf_h(self, obs):
        # Placeholder drift term for kinematic model.
        return 0.0

    def Lg_h(self, obs, act_dim):
        # Placeholder control effectiveness; assume identity.
        return np.ones(act_dim, dtype=np.float32)

    def get_safe_action(self, obs, nominal_action):
        nominal_action = np.asarray(nominal_action, dtype=np.float32)
        act_dim = nominal_action.shape[-1]
        if not self.cvxpy_available:
            return np.clip(nominal_action, -self.act_limit, self.act_limit)

        u = cp.Variable(act_dim)
        u_ref = nominal_action
        h_val = float(self.h(obs))
        Lf = float(self.Lf_h(obs))
        Lg = self.Lg_h(obs, act_dim)

        constraint = Lf + cp.sum(cp.multiply(Lg, u)) + self.alpha * h_val
        constraints = [
            constraint >= 0.0,
            u <= self.act_limit,
            u >= -self.act_limit,
        ]
        objective = cp.Minimize(cp.sum_squares(u - u_ref))
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
        except Exception as e:
            print(f"[Warning] CBF QP solve failed: {e}")
            return np.clip(u_ref, -self.act_limit, self.act_limit)

        if u.value is None:
            return np.clip(u_ref, -self.act_limit, self.act_limit)
        return np.clip(np.array(u.value).squeeze(), -self.act_limit, self.act_limit)
