import pickle
import numpy as np

data = pickle.load(open("models/model_epoch_2000_low_dim_v15_success_0.5-100.pkl", "rb"))
obs, act = data['obs'], data['act']
# print(len(obs[0][57:]))

new_obs = []

for o in obs:
    new_obs.append(o[57:])

pickle.dump(
    {"obs": np.array(new_obs), "act": np.array(act)},
    open(f"models/model_epoch_2000_low_dim_v15_success_0.5-100-linear.pkl", "wb"),
)
