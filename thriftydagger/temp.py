import pickle

f = open("robosuite-30.pkl", "rb")
data = pickle.load(f)
print(data)
