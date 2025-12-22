import pickle
d = pickle.load(open('visualization/angle_100.pkl','rb'))
import numpy as np

print('obs shape', d['obs'].shape, 'act shape', d['act'].shape)
print('sample obs[0]:', d['obs'][:1000:10])
print('sample act[0]:', d['act'][0])