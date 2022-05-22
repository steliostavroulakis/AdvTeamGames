
from custom_gym_4x4 import SimpleEnv
import numpy as np
import random
from libr import *
import matplotlib.pyplot as plt
import os
import sys
import wandb
from libr2 import *
import glob

env = SimpleEnv()

SELECT_ITERATE = 1800

#with open(f'saved_policies/{SELECT_GRID_SIZE}x{SELECT_GRID_SIZE}/player1_table_{SELECT_ITERATE}.npy', 'rb') as f:
#    player1_table = np.load(f)
#with open(f'saved_policies/{SELECT_GRID_SIZE}x{SELECT_GRID_SIZE}/player2_table_{SELECT_ITERATE}.npy', 'rb') as f:
#    player2_table = np.load(f)
#with open(f'saved_policies/{SELECT_GRID_SIZE}x{SELECT_GRID_SIZE}/adv_table_{SELECT_ITERATE}.npy', 'rb') as f:
#    adv_table = np.load(f)

player1_table = play_uniform(env)
player2_table = play_uniform(env)
player3_table = play_uniform(env)
adv_table = play_uniform(env)

env = SimpleEnv()

obs = env.reset()
done = env.done_flag
env.render(0)
if done == True:
    print("done")
iter = 1
while not done:
    action1 = np.random.choice(a=[0,1,2,3,4], p=player1_table[index_table(obs)])
    action2 = np.random.choice(a=[0,1,2,3,4], p=player2_table[index_table(obs)])
    action3 = np.random.choice(a=[0,1,2,3,4], p=player3_table[index_table(obs)])
    action4 = argmax_list(adv_table[index_table(obs)])

    obs, rew, done, info = env.step([action1,action2, action3, action4])
    
    env.render(iter)
    iter+=1

# Create the frames
frames = []
imgs = glob.glob("saved_images/*.png")
imgs = sorted(imgs)
print(imgs)
sys.exit(0)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever

frames[0].save(f'saved_videos/{str(time.time())}.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=500, loop=0)