import pickle, numpy as np

src = "models/offline_dataset_mazeMedium_1000.pkl"  # 你的 1000 檔
dst = "models/offline_dataset_mazeMedium_100.pkl"  # 要輸出的新檔
n = 100

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
