from custom_gym_3x3 import SimpleEnv
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from libr import *

import matplotlib.pyplot as plt
import gym
import os
import wandb
import sys
import wandb
from numpy.random import choice
from libr2 import *

import cv2
import os

image_folder = 'images'
video_name = 'video.avi'

images = [img for img in os.listdir('../videos/') if img.endswith(".png")]
frame = cv2.imread(os.path.join('../videos/', images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join('../videos/', image)))

cv2.destroyAllWindows()
video.release()

sys.exit(0)

print(index_table_inverse(341))
#a = [[0,1,2,3,4],[1,4,2,3,2]]#

#b = one_hot_matrix(a)
#print(b)
sys.exit(0)


env = SimpleEnv()
player1_table = play_nothing(env)
#player1_table = play_uniform(env)
#player1_table = play_random(env)

player2_table = play_nothing(env)
#player2_table = play_uniform(env)
#player2_table = play_random(env)

adv_table = play_uniform(env)

with open('saved_policies/player1_table.npy', 'wb') as f:
    np.save(f, player1_table)

with open('saved_policies/player2_table.npy', 'wb') as f:
    np.save(f, player2_table)

with open('saved_policies/adv_table.npy', 'wb') as f:
    np.save(f, adv_table)
print(adv_table)

sys.exit(0)
env = SimpleEnv()
obs = env.reset()

with open('player1_table_mini.npy', 'rb') as f:
        player1_table_mini = np.load(f)
print(len(player1_table_mini))
#print('Print table index: ',index_table(obs))
plt.imshow(player1_table_mini, cmap='hot', interpolation='nearest')
plt.show()

with open('player2_table_mini.npy', 'rb') as f:
        player2_table_mini = np.load(f)
print(len(player2_table_mini))
#print('Print table index: ',index_table(obs))
plt.imshow(player2_table_mini, cmap='hot', interpolation='nearest')
plt.show()