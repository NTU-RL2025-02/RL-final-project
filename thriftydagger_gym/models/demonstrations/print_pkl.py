import pickle
d = pickle.load(open('new_maze_500.pkl','rb'))
import numpy as np

print('obs shape', d['obs'].shape, 'act shape', d['act'].shape)
print('sample obs[0]:', d['obs'][0])
print('sample act[0]:', d['act'][0])