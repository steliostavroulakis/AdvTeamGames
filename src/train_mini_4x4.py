import copy
import os
import random
import sys

import numpy as np

import wandb
#from custom_gym_3x3 import SimpleEnv
from custom_gym_4x4 import SimpleEnv
from libr4x4 import *
from libr24x4 import *

wandb.init()

# -----------------------------
# -- CREATE AND SAVE TABLES ---
# -----------------------------

env = SimpleEnv()
#print(env.observation_size)
#sys.exit(0)

#player1_table = play_nothing(env)
#print(len(player1_table))
#sys.exit(0)
player1_table = play_uniform(env)
#player1_table = play_random(env)


#player2_table = play_nothing(env)
player2_table = play_uniform(env)
#player2_table = play_random(env)

player3_table = play_uniform(env)

adv_table = play_uniform(env)

with open(f'saved_policies/4x4/player1_table.npy', 'wb') as f:
    np.save(f, player1_table)
with open(f'saved_policies/4x4/player2_table.npy', 'wb') as f:
    np.save(f, player2_table)
with open(f'saved_policies/4x4/player3_table.npy', 'wb') as f:
    np.save(f, player3_table)
with open(f'saved_policies/4x4/adv_table.npy', 'wb') as f:
    np.save(f, plt==adv_table)

# -----------------------------
# -----------------------------
LEARNING_RATE_TEAM = 0.1
GAMMA_TEAM = 0.5
MCMC_TEAM = 100
#---------------------------
num_episodes = 100
max_steps_per_episode = 15
learning_rate = 0.1
discount_rate = 0.5
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 5/num_episodes

#---------------------------
VALUE_ROLLOUTS = 5
VALUE_HORIZON = 0.5

# ----------------------------------------------------------
# -------- Play Policy Gradient VS Best Response -----------
# ----------------------------------------------------------
print(f"Initializing experiment with {env.num_agents} agents in a {env.grid_size}x{env.grid_size} grid world.")

for i in range(20000):
    # ------------------------------------------------------------------------ #
    # ------------------------ ADVERSARY BEST RESPONSE ----------------------- #
    # ------------------------------------------------------------------------ #
    adv_table_prev = copy.deepcopy(adv_table)
    player1_table_prev = copy.deepcopy(player1_table)
    player2_table_prev = copy.deepcopy(player2_table)
    player3_table_prev = copy.deepcopy(player3_table)

    print("Iter ",i,": Adversary Best Responding... ")

    #print(value(env, player1_table, player2_table, adv_table, VALUE_ROLLOUTS, VALUE_HORIZON)[2])
    adv_table = best_response(num_episodes, max_steps_per_episode, learning_rate, discount_rate, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, env, player1_table, player2_table, player3_table,adv_table)
    
    #print(value(env, player1_table, player2_table, adv_table, VALUE_ROLLOUTS, VALUE_HORIZON)[2])
    adv_value = value(env, player1_table, player2_table, player3_table, adv_table, VALUE_ROLLOUTS, VALUE_HORIZON)[2]
    wandb.log({'Value Adversary':adv_value}, commit=False)
    wandb.log({"Adversary - Policy Difference":frob_matrix_norm(one_hot_matrix(adv_table_prev),one_hot_matrix(adv_table))}, commit=False)
    # ------------------------------------------------------------------------ #
    # ------------------------ ADVERSARY BEST RESPONSE ----------------------- #
    # ------------------------------------------------------------------------ #
    print("        Team doing 1 PG iteration...")
    
    # CALCULATE GRADS OF PLAYERS
    grad_est_1 = np.zeros_like(player1_table) 
    grad_est_2 = np.zeros_like(player2_table) 
    grad_est_3 = np.zeros_like(player3_table) 

    for _ in range(MCMC_TEAM):
        obs = env.reset()
        traj_1 = get_mini_trajectory(env, obs, player1_table, player2_table, player3_table, adv_table, GAMMA_TEAM)
        grad_est_1 += grad_estim_iter(0, player1_table, traj_1)
    grad_est_1 = grad_est_1/MCMC_TEAM

    for _ in range(MCMC_TEAM):
        obs = env.reset()
        traj_2 = get_mini_trajectory(env, obs, player1_table, player2_table, player3_table, adv_table, GAMMA_TEAM)
        grad_est_2 += grad_estim_iter(1, player2_table, traj_2)
    grad_est_2 = grad_est_2/MCMC_TEAM

    for _ in range(MCMC_TEAM):
        obs = env.reset()
        traj_3 = get_mini_trajectory(env, obs, player1_table, player2_table, player3_table, adv_table, GAMMA_TEAM)
        grad_est_3 += grad_estim_iter(1, player3_table, traj_3)
    grad_est_3 = grad_est_3/MCMC_TEAM

    # UPDATE POLICIES OF PLAYERS-
    player1_table = player1_table + grad_est_1*LEARNING_RATE_TEAM
    for row in range(len(player1_table)):
        player1_table[row] = projsplx(player1_table[row])
    
    player2_table = player2_table + grad_est_2*LEARNING_RATE_TEAM
    for row in range(len(player2_table)):
        player2_table[row] = projsplx(player2_table[row])

    player3_table = player3_table + grad_est_3*LEARNING_RATE_TEAM
    for row in range(len(player3_table)):
        player3_table[row] = projsplx(player3_table[row])

    # LOG POLICIES AND VALUES
    team_value = value(env, player1_table, player2_table, adv_table, VALUE_ROLLOUTS, VALUE_HORIZON)
    wandb.log({'Value Player 1':team_value[0]},commit=False)
    wandb.log({'Value Player 2':team_value[1]},commit=False)
    wandb.log({'Value Player 3':team_value[2]},commit=False)

    wandb.log({'Policy 1 Frob Norm':frob_matrix_norm(player1_table_prev,player1_table)},commit=False)
    wandb.log({'Policy 2 Frob Norm':frob_matrix_norm(player2_table_prev,player2_table)},commit=False)
    wandb.log({'Policy 3 Frob Norm':frob_matrix_norm(player3_table_prev,player3_table)})

    if i % 500 == 0:

        with open(f'saved_policies/4x4/player1_table_{i}.npy', 'wb') as f:
            np.save(f, player1_table)

        with open(f'saved_policies/4x4/player2_table_{i}.npy', 'wb') as f:
            np.save(f, player2_table)

        with open(f'saved_policies/4x4/player3_table_{i}.npy', 'wb') as f:
            np.save(f, player3_table)

        adv_table = best_response(num_episodes, max_steps_per_episode, learning_rate, discount_rate, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, env, player1_table, player2_table, adv_table)
        with open(f'saved_policies/4x4/adv_table_{i}.npy', 'wb') as f:
            np.save(f, adv_table)
