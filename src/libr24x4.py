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

def play_nothing(env):
    tmp = np.zeros((env.grid_size**env.observation_size,env.num_actions))
    for row in range(len(tmp)):
        tmp[row] = [1,0,0,0,0]
    return tmp

def play_uniform(env):
    tmp = np.ones((env.grid_size**env.observation_size,env.num_actions))
    tmp = tmp/np.sum(tmp,axis=1)[:,None]
    return tmp

def play_random(env):
    return np.random.rand(env.grid_size**env.observation_size,env.num_actions)

def one_hot_matrix(mat):
    mat1 = mat.copy()
    for row in range(len(mat1)):
        max_index = np.argmax(mat1[row])
        mat1[row] = [0,0,0,0,0]
        mat1[row][max_index] = 1
    return mat1

def argmax_list(lst):
    return np.argmax(lst)
