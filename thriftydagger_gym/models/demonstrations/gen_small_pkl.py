import pickle, numpy as np

src = "offline_data_100.pkl"  
dst = "offline_data_50.pkl"  
n = 50

data = pickle.load(open(src, "rb"))
num = len(data["obs"])
idx = np.random.choice(num, size=n, replace=False)

sub = {
    "obs": data["obs"][idx],
    "act": data["act"][idx],
}
with open(dst, "wb") as f:
    pickle.dump(sub, f)
print(f"Saved {n} samples to {dst}")
