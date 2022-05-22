import copy
from operator import index
import os
import random
import sys
import numpy as np
import wandb
from custom_gym_3x3 import SimpleEnv
from libr import *
from libr2 import *
from numpy import linalg as LA

# -----------------------------
# -- CREATE AND SAVE TABLES ---
# -----------------------------
env = SimpleEnv()

#player1_table = play_nothing(env)
player1_table = play_uniform(env)
#player1_table = play_random(env)

#player2_table = play_nothing(env)
player2_table = play_uniform(env)
#player2_table = play_random(env)

adv_table = play_uniform(env)
#adv_table = play_nothing(env)

# -----------------------------
# - INITIALIZE HYPERPARAMETERS-
# -----------------------------
LEARNING_RATE_TEAM = 0.2
GAMMA_TEAM = 0.9
MCMC_TEAM = 2000
#---------------------------
num_episodes = 40000
max_steps_per_episode = 10
learning_rate = 0.1
discount_rate = 0.66
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 5/num_episodes

conf_dict = {"POLG - LR":LEARNING_RATE_TEAM,
				"MCMC - CUTOFF":GAMMA_TEAM,
				"MCMC - SAMPLES":MCMC_TEAM,
				"QLRN - EPISODES": num_episodes,
				"QLRN - CUTOFF": max_steps_per_episode,
				"QLRN - LR": learning_rate,
                "INIT_DIST - TEAM":"UNIFORM",
}

wandb.init(project="Adversarial Team Games",config=conf_dict)

# ----------------------------------------------------------
# -------- Play Policy Gradient VS Best Response -----------
# ----------------------------------------------------------
print(f"Initializing experiment with {env.num_agents} agents in a {env.grid_size}x{env.grid_size} grid world.")

for i in range(20000):
    
    adv_table = play_uniform(env)
    adv_table_prev = copy.deepcopy(adv_table)
    player1_table_prev = copy.deepcopy(player1_table)
    player2_table_prev = copy.deepcopy(player2_table)
    # ------------------------------------------------------------------------ #
    # ------------------------ ADVERSARY BEST RESPONSE ----------------------- #
    # ------------------------------------------------------------------------ #
    adv_table = best_response(num_episodes, max_steps_per_episode, learning_rate, discount_rate, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, env, player1_table, player2_table,adv_table)
    value_adv = adversary_value(adv_table)
    wandb.log({"Adversary Frob Norm":frob_matrix_norm(one_hot_matrix(adv_table_prev),one_hot_matrix(adv_table))}, commit=False)
    wandb.log({"Adversary Value":value_adv, "Team Value":-value_adv},commit=False)
    # ------------------------------------------------------------------------ #
    # ------------------------ ADVERSARY BEST RESPONSE ----------------------- #
    # ------------------------------------------------------------------------ #
    print("Iter ",i,": Team doing 1 PG iteration...")

    # CALCULATE GRADS OF PLAYERS
    grad_est_1 = np.zeros_like(player1_table) 
    grad_est_2 = np.zeros_like(player2_table) 
    for _ in range(MCMC_TEAM):
        obs = env.reset()
        traj_1 = get_mini_trajectory(env, obs, player1_table, player2_table, adv_table, GAMMA_TEAM)
        grad1 = grad_estim_iter(0, player1_table, traj_1, env.num_actions)
        grad_est_1 += grad1
    grad_est_1 = grad_est_1/MCMC_TEAM

    for _ in range(MCMC_TEAM):
        obs = env.reset()
        traj_2 = get_mini_trajectory(env, obs, player1_table, player2_table, adv_table, GAMMA_TEAM)
        grad2 = grad_estim_iter(1, player2_table, traj_2, env.num_actions)
        grad_est_2 += grad2
    grad_est_2 = grad_est_2/MCMC_TEAM

    # UPDATE POLICIES OF PLAYERS
    player1_table = player1_table + grad_est_1*LEARNING_RATE_TEAM
    for row in range(len(player1_table)):
        player1_table[row] = projsplx(player1_table[row])
    
    player2_table = player2_table + grad_est_2*LEARNING_RATE_TEAM
    for row in range(len(player2_table)):
        player2_table[row] = projsplx(player2_table[row])

    pol_diff_1 = frob_matrix_norm(player1_table_prev,player1_table)
    pol_diff_2 = frob_matrix_norm(player2_table_prev,player2_table)
    wandb.log({'Policy 1 Frob Norm':pol_diff_1, 'Policy 2 Frob Norm':pol_diff_2})

    if i % 500 == 0:

        with open(f'saved_policies/{env.grid_size}x{env.grid_size}/player1_table_{i}.npy', 'wb') as f:
            np.save(f, player1_table)

        with open(f'saved_policies/{env.grid_size}x{env.grid_size}/player2_table_{i}.npy', 'wb') as f:
            np.save(f, player2_table)

        adv_table = play_uniform(env)
        adv_table = best_response(num_episodes, max_steps_per_episode, learning_rate, discount_rate, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, env, player1_table, player2_table,adv_table)
        with open(f'saved_policies/{env.grid_size}x{env.grid_size}/adv_table_{i}.npy', 'wb') as f:
            np.save(f, adv_table)