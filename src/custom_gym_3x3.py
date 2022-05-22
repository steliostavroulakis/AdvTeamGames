from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from libr import *
from gym import logger, spaces
import sys
import time 

class SimpleEnv(Env):

    def __init__(self):
        
        # Environment Specs
        self.num_agents = 2
        self.num_actions = 5
        self.grid_size = 4

        # Goal Location
        self.target1_x = 0 # random.randint(0, self.grid_size-1)
        self.target1_y = 0 # random.randint(0, self.grid_size-1)
        self.target2_x = self.grid_size-1 # random.randint(0, self.grid_size-1)
        self.target2_y = self.grid_size-1 # random.randint(0, self.grid_size-1)

        # Auxilary reward multiplier
        self.aux_multiplier = 0.01

        # Player Summoning
        self.agent_dict = dict()
        for player_int in range(self.num_agents):
            self.agent_dict['player'+str(player_int)] = TeamPlayer(self.grid_size)
        self.agent_dict['player'+str(self.num_agents)] = AdvPlayer(self.grid_size)

        if self.team_wins() or self.adv_wins():
            self.done_flag = True
        else:
            self.done_flag = False

        self.observation_size = self.obs()
        self.observation_size = len(self.obs())

    def step(self, action_list):

        self.done_flag = False
        
        # Move agents in environment
        for act in range(len(action_list)):
            self.agent_dict['player'+str(act)].move(action_list[act])

        reward_list = [0,0,0]
        #aux_rew = self.aux_reward()*self.aux_multiplier
        #print('Aux Reward = ',-aux_rew)
        #reward_list = [-aux_rew, -aux_rew, aux_rew]

        if self.team_wins() and self.adv_wins():
            reward_list = [0,0,0]
            self.done_flag = True

        if self.team_wins():
            reward_list = [1,1,-1]
            self.done_flag = True
        
        if self.adv_wins():
            reward_list = [-1,-1,1]
            self.done_flag = True
        
        return self.obs(), reward_list, self.done_flag, {}

    def render(self,iter):
        self.print_coords()
        #print("Max Distance from target = ",self.aux_reward())
        fig,ax = plt.subplots(1)
        fig.subplots_adjust(bottom=0.3)

        colorstate = np.zeros((self.grid_size, self.grid_size,3), dtype=np.uint8)
        colorstate[self.target1_x][self.target1_y] = (55, 255, 55)
        colorstate[self.target2_x][self.target2_y] = (55, 255, 55)
        #colorstate[self.target2_x][self.target2_y] = (55, 255, 55)
        
        for player_int in range(self.num_agents):
            
            colorstate[self.agent_dict['player'+str(player_int)].x][self.agent_dict['player'+str(player_int)].y] = (150, 55, 255)
        colorstate[self.agent_dict['player'+str(self.num_agents)].x][self.agent_dict['player'+str(self.num_agents)].y] = (255, 50, 50)
        ax.imshow(colorstate, interpolation='None')
        ax.axis('off')

        for player in range(self.num_agents):
            x_coord = self.agent_dict['player'+str(player)].x
            y_coord = self.agent_dict['player'+str(player)].y
            fig.text(0.3,0.25-0.05*player,f'Player {player}: ({x_coord},{y_coord})')
        x_coord_adv = self.agent_dict['player'+str(self.num_agents)].x
        y_coord_adv = self.agent_dict['player'+str(self.num_agents)].y
        fig.text(0.3,0.15,f'Adversary: ({x_coord_adv},{y_coord_adv})')
        fig.text(0.3,0.10,f'Targets: ({self.target1_x},{self.target1_y}) and ({self.target2_x},{self.target2_y})')
        fig.text(0.55,0.25, f'Iteration: {iter}')
        plt.savefig(f'saved_images/out_{iter}.png')
        plt.show()

    def reset(self):

        # Player Summoning
        self.agent_dict = dict()
        for player_int in range(self.num_agents):
            self.agent_dict['player'+str(player_int)] = TeamPlayer(self.grid_size)
        self.agent_dict['player'+str(self.num_agents)] = AdvPlayer(self.grid_size)

        if self.team_wins() or self.adv_wins():
            self.done_flag = True
        else:
            self.done_flag = False

        return self.obs()
    
    def staged_reset(self, coords):

        for player_int in range(self.num_agents):
            self.agent_dict['player'+str(player_int)] = TeamPlayer(self.grid_size)
            self.agent_dict['player'+str(player_int)].x = coords[2*player_int]
            self.agent_dict['player'+str(player_int)].y = coords[2*player_int+1]
        self.agent_dict['player'+str(self.num_agents)] = AdvPlayer(self.grid_size)
        self.agent_dict['player'+str(self.num_agents)].x = coords[2*self.num_agents]
        self.agent_dict['player'+str(self.num_agents)].y = coords[2*self.num_agents+1]

        if self.team_wins() or self.adv_wins():
            self.done_flag = True
        else:
            self.done_flag = False

        return self.obs()

    def obs(self):
        observation_space = []
        for player_int in range(self.num_agents):
            observation_space.append(self.agent_dict['player'+str(player_int)].x)
            observation_space.append(self.agent_dict['player'+str(player_int)].y)
        observation_space.append(self.agent_dict['player'+str(self.num_agents)].x)
        observation_space.append(self.agent_dict['player'+str(self.num_agents)].y)
        return observation_space # [x,y]
    
    def aux_reward(self):
        obs_list = self.obs()
        #print(obs_space_sample)num_agents
        d1 = abs(obs_list[0]-self.target1_x) + abs(obs_list[1]-self.target1_y)
        d2 = abs(obs_list[0]-self.target2_x) + abs(obs_list[1]-self.target2_y)
        d3 = abs(obs_list[2]-self.target1_x) + abs(obs_list[3]-self.target1_y)
        d4 = abs(obs_list[2]-self.target2_x) + abs(obs_list[3]-self.target2_y)

        min_target_1 = min(d1,d3)
        min_target_2 = min(d2,d4)

        return min_target_1 + min_target_2

    def print_coords(self):
        for player in range(self.num_agents):
            x_coord = self.agent_dict['player'+str(player)].x
            y_coord = self.agent_dict['player'+str(player)].y
            print(f'Player {player}: ({x_coord},{y_coord})')
        print('--------------')
        x_coord_adv = self.agent_dict['player'+str(self.num_agents)].x
        y_coord_adv = self.agent_dict['player'+str(self.num_agents)].y
        print(f'Adversary: ({x_coord_adv},{y_coord_adv})')
        print(f'Targets: ({self.target1_x},{self.target1_y}) and ({self.target2_x},{self.target2_y})')
        print('--------------')
 
    def team_wins(self):

        target1_flag = False
        target2_flag = False

        for team_agent in range(self.num_agents):

            x = 0
            y = 1
            coords = [self.agent_dict['player'+str(team_agent)].x, self.agent_dict['player'+str(team_agent)].y]

            if coords[x] == self.target1_x and coords[y] == self.target1_y:
                target1_flag = True
            if coords[x] == self.target2_x and coords[y] == self.target2_y:
                target2_flag = True
            
        if target1_flag == True and target2_flag == True:
            return True
        else:
            return False

    def adv_wins(self):

        x=0
        y=1
        coords = [self.agent_dict['player'+str(self.num_agents)].x, self.agent_dict['player'+str(self.num_agents)].y]

        if coords[x] == self.target1_x and coords[y] == self.target1_y:
            return True
        if coords[x] == self.target2_x and coords[y] == self.target2_y:
            return True
        
        return False