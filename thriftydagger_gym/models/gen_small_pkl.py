import pickle, numpy as np

src = "models/offline_dataset_rulebased_10000.pkl"  
dst = "models/offline_dataset_rulebased_1000.pkl"  
n = 1000

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
