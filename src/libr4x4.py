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

def frob_matrix_norm(mat1, mat2):
    return LA.norm(mat1-mat2,'fro')

def nuc_matrix_norm(mat1, mat2):
    return LA.norm(mat1-mat2,'nuc')

def base_convert(i, b):
    result = []
    while i > 0:
            result.insert(0, i % b)
            i = i // b
    return result

def index_table(lst):
    base = 4
    q_table_size_aux = lst
    s = [str(integer) for integer in q_table_size_aux]
    a_string = "".join(s)
    a_int = int(a_string)
    return sum([int(character) * base ** index for index,character in enumerate(str(a_int)[::-1])])

def index_table_inverse(i):
    result = []
    while i > 0:
            result.insert(0, i % 4)
            i = i // 4
    while len(result)<8:
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

def grad_estim_iter(player, player_table, trajectory):
    
    epsilon = 1e-9

    actions = trajectory['actions'][player] # [0,1,2,3,2,1]
    rewards = trajectory['rewards'][player]
    states = trajectory['states']

    #DiscountedReturns = []
    #for t in range(len(rewards)):
    #    G = 0.0
    #    for k, r in enumerate(rewards[t:]):
    #        G += (gamma**k)*r
    #    DiscountedReturns.append(G)

    R = np.sum(rewards) # 100
    g = np.zeros_like(player_table)
    for s,a, in zip(states, actions):
        g[s,a] += 1
    normalized_policy = player_table*(1-epsilon) + epsilon/5
    g = g*(1/(normalized_policy))
    return R*g

def get_mini_trajectory(environment, obs, player1_table, player2_table, player3_table, adv_table, horizon):

        CUTOFF_STEPS = np.random.geometric(p=1-horizon, size=1)
        #print('CUTOFF STEPS = ',CUTOFF_STEPS)

        obs = environment.staged_reset(obs) # [x,y]

        player1_state = index_table(obs)
        player2_state = index_table(obs) 
        player3_state = index_table(obs)
        adv_state = index_table(obs)

        done = environment.done_flag
        #print("inner loop",done)
        States = []
        Actions1, Rewards1 = [],[]
        Actions2, Rewards2 = [],[]
        Actions3, Rewards3 = [],[]
        Actions4, Rewards4 = [],[]

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

            probs3 = player3_table[player3_state] 
            probs3 /= probs3.sum()
            action3 = choice(range(len(probs3)), p=probs3) # 3
            action_list.append(action3)  

            action4 = np.argmax(adv_table[adv_state])
            action_list.append(action4)  

            States.append(player1_state) # [1,2]

            Actions1.append(action1) # 2
            Actions2.append(action2) # 2
            Actions3.append(action3) # 2
            Actions4.append(action4) # 2

            new_state , rew, done, _ = environment.step(action_list)

            player1_state = index_table(new_state)
            player2_state = index_table(new_state)
            player3_state = index_table(new_state)
            adv_state = index_table(new_state)

            Rewards1.append(rew[0])  # 0
            Rewards2.append(rew[1])
            Rewards3.append(rew[2])  # 0
            Rewards3.append(rew[3])  # 0


            if done:
                continue

        traj_dict = {'states': States,
                     'actions': [Actions1, Actions2, Actions3, Actions4],
                     'rewards': [Rewards1, Rewards2, Rewards3, Rewards4],
                     'steps': len(States)}

        return traj_dict

def best_response(num_episodes, max_steps_per_episode, learning_rate, discount_rate, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, env, player1_table, player2_table, player3_table, adv_table):

    #state = 1
    #action = 1
    #print(adv_table[state,action])
    #sys.exit(0)
    rewards_all_episodes = []

    # Q-learning algorithm
    for episode in range(num_episodes):
        state = env.reset() # [1,2,3,4,5,6]
        #print(state)
        #sys.exit(0)
        state_num = index_table(state)
        player1_state = state_num # 4096
        player2_state = state_num # 4096
        player3_state = state_num
        adv_state = state_num # 4096

        done = False
        rewards_current_episode = 0
        
        for _ in range(max_steps_per_episode):   
            
            action_list = []

            probs1 = player1_table[player1_state] # distribution
            probs1 /= probs1.sum()
            action1 = choice(range(len(probs1)),p=probs1)
            action_list.append(action1) 

            probs2 = player2_table[player2_state]
            probs2 /= probs2.sum()
            action2 = choice(range(len(probs2)),p=probs2)
            action_list.append(action2) 

            probs3 = player3_table[player3_state]
            probs3 /= probs3.sum()
            action3 = choice(range(len(probs3)),p=probs3)
            action_list.append(action3) 

            # Exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action_list.append(np.argmax(adv_table[adv_state,:]))
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
            player1_state = state_num # 4096
            player2_state = state_num # 4096
            adv_state = state_num # 4096
            rewards_current_episode += reward      
            
            if done == True: 
                break
            
        # Exploration rate decay
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)    
        
        rewards_all_episodes.append(rewards_current_episode)

    return adv_table

def value(env, player1_table, player2_table, player3_table, adv_table, VALUE_ROLLOUTS, VALUE_HORIZON):

    sum_val1 = []
    sum_val2 = []
    sum_val3 = []
    sum_val4 = []

    for st in range(len(player1_table)):
        obs = index_table_inverse(st)
        for rollouts in range(VALUE_ROLLOUTS):
            traj = get_mini_trajectory(env, obs, player1_table, player2_table, player3_table, adv_table, VALUE_HORIZON)
            for _ in range(3):
                disc_rew1 = get_discounted_rewards(traj['rewards'][0],VALUE_HORIZON)
                sum_val1.append(disc_rew1)

                disc_rew2 = get_discounted_rewards(traj['rewards'][1],VALUE_HORIZON)
                sum_val2.append(disc_rew2)

                disc_rew3 = get_discounted_rewards(traj['rewards'][2],VALUE_HORIZON)
                sum_val3.append(disc_rew3)

                disc_rew4 = get_discounted_rewards(traj['rewards'][3],VALUE_HORIZON)
                sum_val4.append(disc_rew4)

    return sum(sum_val1)/len(sum_val1), sum(sum_val2)/len(sum_val2), sum(sum_val3)/len(sum_val3), sum(sum_val4)/len(sum_val4)

def get_discounted_rewards(rew_list, DISCOUNT):
    DiscountedReturns = []
    for t in range(len(rew_list)):
        G = 0.0
        for k, r in enumerate(rew_list[t:]):
            G += (DISCOUNT**k)*r
        DiscountedReturns.append(G)

    return sum(DiscountedReturns)

class TeamPlayer:
    def __init__(self,grid_size):
        self.grid_size = grid_size
        self.reward = 0
        self.x = random.randint(0, grid_size-1) #grid_size-1
        self.y = random.randint(0, grid_size-1) #grid_size-1
    
    def move(self,direction):
        if direction == 0: # no movement
            self.x = self.x
            self.y = self.y

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
        if direction == 0: # no movement
            self.x = self.x
            self.y = self.y

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