from os import environ
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from scipy.spatial.distance import jensenshannon
import numpy as np
from numpy.random import choice
import sys
from numpy import linalg as LA
import plotly.express as px
from scipy.special import rel_entr

def projsplx(y):
    """Python implementation of:
    https://arxiv.org/abs/1101.6081"""
    s = np.sort(y)
    n = len(y) ; flag = False
    
    parsum = 0
    tmax = -np.inf
    for idx in range(n-2, -1, -1):
        parsum += s[idx+1]
        tmax = (parsum - 1) / (n - (idx + 1) )
        if tmax >= s[idx]:
            flag = True ; break
    
    if not flag:
        tmax = (np.sum(s) - 1) / n
    
    return np.maximum(y - tmax, 0)

def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    #plt.title("%s | Step: %d %s" % (env._spec.id,step, info))
    plt.axis('off')

def base_convert(i, b):
    result = []
    while i > 0:
            result.insert(0, i % b)
            i = i // b
    return result

def index_table(lst):
    grid_size = 4

    base = grid_size
    q_table_size_aux = lst
    s = [str(integer) for integer in q_table_size_aux]
    a_string = "".join(s)
    a_int = int(a_string)
    return sum([int(character) * base ** index for index,character in enumerate(str(a_int)[::-1])])

def index_table_inverse(num):
    grid_size = 4

    result = []
    while num > 0:
            result.insert(0, num % grid_size)
            num = num // grid_size
    while len(result)<6:
      result.insert(0,0)
    return result

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def dist_distr(list1, list2):
    return jensenshannon(list1,list2)

def grad_estim_iter(player, player_table, trajectory, action_space):
    
    epsilon = 1e-15
    actions = trajectory['actions'][player]
    rewards = trajectory['rewards']
    states = trajectory['states']

    R = np.sum(rewards)
    #print('sum of rewards = ',R)
    g = np.zeros_like(player_table)
    for s,a, in zip(states, actions):
        g[s,a] += 1
        #print(f"adding 1 to state {s}")
        #print(f" g[{s},{a}] = ",g[s,a])
    normalized_policy = player_table*(1-epsilon) + epsilon/action_space
    g = g*(1/(normalized_policy))
    return R*g

def get_mini_trajectory(environment, obs, player1_table, player2_table,adv_table, horizon):

        CUTOFF_STEPS = np.random.geometric(p=1-horizon, size=1)
        #print('CUTOFF STEPS = ',CUTOFF_STEPS)

        obs = environment.staged_reset(obs) # [x,y]

        player1_state = index_table(obs)
        player2_state = index_table(obs) 
        adv_state = index_table(obs)

        done = environment.done_flag
        #print("inner loop",done)
        States = []
        Actions1 = []
        Actions2 = []
        Rewards = []

        # GATHER ONE EPISODE OF EXPERIENCE
        iter = 0
        while not environment.done_flag and iter <= CUTOFF_STEPS:

            iter+=1

            action_list = [] #[]
            
            probs1 = player1_table[player1_state] 
            probs1 /= probs1.sum()
            action1 = choice(range(len(probs1)), p=probs1) # 3
            action_list.append(action1)                    # [3]

            probs2 = player2_table[player2_state] 
            probs2 /= probs2.sum()
            action2 = choice(range(len(probs2)), p=probs2) # 3
            action_list.append(action2)                    # [3]

            action3 = np.random.choice(np.flatnonzero(adv_table[index_table(obs)] == adv_table[index_table(obs)].max()))
            action_list.append(action3)  

            States.append(player1_state) # [1,2]

            Actions1.append(action1) # 2
            Actions2.append(action2) # 2

            new_state , rew, done, _ = environment.step(action_list)

            player1_state = index_table(new_state)
            player2_state = index_table(new_state)

            Rewards.append(rew[0])  # 0

            if done:
                continue

        traj_dict = {'states': States,
                     'actions': [Actions1, Actions2],
                     'rewards': Rewards,
                     'steps': len(States)}
        return traj_dict

def best_response(num_episodes, max_steps_per_episode, learning_rate, discount_rate, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, env, player1_table, player2_table, adv_table):

    rewards_all_episodes = []

    # Q-learning algorithm
    for episode in range(num_episodes):
        state = env.reset()
        state_num = index_table(state)
        player1_state = state_num 
        player2_state = state_num 
        adv_state = state_num 

        done = env.done_flag
        rewards_current_episode = 0
        
        for _ in range(max_steps_per_episode):   

            if done == True: 
                break
            
            action_list = []

            probs1 = player1_table[player1_state]
            probs1 /= probs1.sum()
            action1 = choice(range(len(probs1)),p=probs1)
            action_list.append(action1) 

            probs2 = player2_table[player2_state]
            probs2 /= probs2.sum()
            action2 = choice(range(len(probs2)),p=probs2)
            action_list.append(action2) 


            # Exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action_list.append(np.random.choice(np.flatnonzero(adv_table[adv_state,:] == adv_table[adv_state,:].max())))
            else:
                action_list.append(np.random.randint(4))

            new_state, reward, done, info = env.step(action_list)

            action = action_list[-1]
            reward = reward[-1]
            new_state_num = index_table(new_state)

            adv_table[state_num, action] = adv_table[state_num, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(adv_table[new_state_num, :]))
            
            state = new_state
            state_num = index_table(state)
            player1_state = state_num
            player2_state = state_num
            adv_state = state_num
            rewards_current_episode += reward      
            
        # Exploration rate decay
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)    
        
        rewards_all_episodes.append(rewards_current_episode)

    return adv_table

def adversary_value(adv_table):
    v_row = []
    for q_row in adv_table:
        v_row.append(max(q_row))
    return sum(v_row)/len(v_row)

class TeamPlayer:
    def __init__(self,grid_size):
        self.grid_size = grid_size
        self.reward = 0
        self.x = random.randint(0, grid_size-1) #grid_size-1
        self.y = random.randint(0, grid_size-1) #grid_size-1
    
    def move(self,direction):

        if direction == 0: # move up
            pass

        elif direction == 1: # move up
            if self.x <=0:
                self.x = self.x
            else:
                self.x -=1

        elif direction == 2: # move down
            if self.x >=self.grid_size-1:
                self.x = self.x
            else:
                self.x +=1

        elif direction == 3: # move right
            if self.y >=self.grid_size-1:
                self.y = self.y
            else:
                self.y +=1

        elif direction == 4: # move left
            if self.y <= 0:
                self.y = self.y
            else:
                self.y -=1

class AdvPlayer:
    def __init__(self,grid_size):
        self.grid_size = grid_size
        self.reward = 0
        self.x = random.randint(0, grid_size-1)
        self.y = random.randint(0, grid_size-1)

    def move(self,direction):
         
        if direction == 0: # NoP
            pass

        elif direction == 1: # move up
            if self.x <=0:
                self.x = self.x
            else:
                self.x -=1

        elif direction == 2: # move down
            if self.x >=self.grid_size-1:
                self.x = self.x
            else:
                self.x +=1

        elif direction == 3: # move right
            if self.y >=self.grid_size-1:
                self.y = self.y
            else:
                self.y +=1

        elif direction == 4: # move left
            if self.y <= 0:
                self.y = self.y
            else:
                self.y -=1