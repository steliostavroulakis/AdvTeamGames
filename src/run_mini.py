from custom_gym_3x3 import SimpleEnv
import numpy as np
import random
from libr import *
import matplotlib.pyplot as plt
import os
import sys
import wandb
from libr2 import *
from PIL import Image
import glob
import os
 
dir_to_delete = 'saved_images'
for f in os.listdir(dir_to_delete):
    os.remove(os.path.join(dir_to_delete, f))
 

SELECT_ITERATE = 500
SELECT_GRID_SIZE = 4

with open(f'saved_policies/{SELECT_GRID_SIZE}x{SELECT_GRID_SIZE}/player1_table_{SELECT_ITERATE}.npy', 'rb') as f:
    player1_table = np.load(f)
with open(f'saved_policies/{SELECT_GRID_SIZE}x{SELECT_GRID_SIZE}/player2_table_{SELECT_ITERATE}.npy', 'rb') as f:
    player2_table = np.load(f)
with open(f'saved_policies/{SELECT_GRID_SIZE}x{SELECT_GRID_SIZE}/adv_table_{SELECT_ITERATE}.npy', 'rb') as f:
    adv_table = np.load(f)


num_episodes = 40000
max_steps_per_episode = 10
learning_rate = 0.1
discount_rate = 0.66
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 5/num_episodes

env = SimpleEnv()
plt.imshow(adv_table[56:122])
plt.show()
sys.exit(0)
adv_table = best_response(num_episodes, max_steps_per_episode, learning_rate, discount_rate, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, env, player1_table, player2_table,adv_table)
plt.imshow(adv_table[56:122])
plt.show()
#sys.exit(0)

#plt.imshow(player1_table[index_table([0,1,3,3,0,0]):index_table([0,1,3,3,3,3])])
#plt.colorbar()
#plt.show()
#sys.exit(0)

env = SimpleEnv()
#adv_table = play_nothing(env)
obs = env.staged_reset([1,1,2,2,2,1])
env.render(0)
#env.render(2)
#sys.exit(0)
done = env.done_flag

i=1
while not done:
    action1 = np.random.choice(a=[0,1,2,3,4], p=player1_table[index_table(obs)])
    action2 = np.random.choice(a=[0,1,2,3,4], p=player2_table[index_table(obs)])
    print(adv_table[index_table(obs)])
    action3 = np.random.choice(np.flatnonzero(adv_table[index_table(obs)] == adv_table[index_table(obs)].max()))
    print(action3)
    obs, rew, done, info = env.step([action1,action2, action3])
    
    env.render(i)
    i+=1

 
# Create the frames
frames = []
imgs = glob.glob("saved_images/*.png")
imgs = sorted(imgs)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
if len(frames)>0:
    frames[0].save(f'saved_videos/{str(time.time())}.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=500, loop=0)