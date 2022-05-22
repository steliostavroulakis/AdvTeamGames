from calendar import c
from custom_gym_3x3 import SimpleEnv
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from libr import *
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import gym
import os
import wandb
import sys
from numpy.random import choice
from libr2 import *

import cv2
import os
import time 
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib


wandb.init(project='test')

for i in range(1000):
    pol_diff_1 = i
    pol_diff_2 = 2*i
    wandb.log({'Policy Frob Norms':(pol_diff_1, pol_diff_2)})


sys.exit(0)

matplotlib.rcParams.update({'font.size': 14})

# Read the CSV into a pandas data frame (df)
#   With a df you can do many things
#   most important: visualize data with Seaborn
values = pd.read_csv('value_fresh.csv', delimiter=',')
print(values.head())
step = values['Step'].tolist()
val_avg = values['avg'].tolist()

policy2 = pd.read_csv('policy2.csv', delimiter=',')
print(policy2.head())
step = policy2['Step'].tolist()
p2_avg = policy2['avg_values'].tolist()
p2_min = policy2['min_values'].tolist()
p2_max = policy2['max_values'].tolist()


policy1 = pd.read_csv('policy1.csv', delimiter=',')
print(policy1.head())
step = policy1['Step'].tolist()
p1_avg = policy1['avg_values'].tolist()
p1_min = policy1['min_values'].tolist()
p1_max = policy1['max_values'].tolist()

fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.plot(step,gaussian_filter1d(p1_avg,12), c='darkslateblue')
plt.plot(step,gaussian_filter1d(p2_avg,12), c='darkorange')
plt.xlabel('Iterations')
plt.ylabel('Frobenius norm of joint policies')
plt.savefig('policies', dpi=200)


fig, ax = plt.subplots()
#plt.plot(gaussian_filter1d(p1_min,12))
#plt.plot(gaussian_filter1d(p1_max,12))

norm_value = [x / 4 for x in val_avg]
ax.plot(gaussian_filter1d(norm_value,12))
plt.xlabel('Iterations')
plt.ylabel('Team Value')
plt.axhline(y = 0,c='0', lw=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#plt.show()
plt.savefig('value', dpi=200)
#print(a)
sys.exit(0)
# Or export it in many ways, e.g. a list of tuples
tuples = [tuple(x) for x in df.values]

# or export it as a list of dicts
dicts = df.to_dict().values()
print(dicts)