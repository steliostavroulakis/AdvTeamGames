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
import os

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
        max_index = np.random.choice(np.flatnonzero(mat1[row] == mat1[row].max()))
        mat1[row] = [0,0,0,0,0]
        mat1[row][max_index] = 1
    return mat1

def frob_matrix_norm(mat1, mat2):
    return LA.norm(mat1-mat2,'fro')

def max_matrix_norm(mat1, mat2):
    mat = mat1 - mat2
    #print(mat)
    max_val = 0
    for row in range(len(mat)):
        for col in range(len(mat[row])):
            if abs(mat[row,col]) > max_val:
                max_val = abs(mat[row,col])
    return max_val

def nuc_matrix_norm(mat1, mat2):
    return LA.norm(mat1-mat2,'nuc')

def kl_norm(mat1, mat2):
    suma = []
    for i in range(len(mat1)):
        suma.append(sum(rel_entr(mat1[i], mat2[i])))
    return sum(suma)/len(suma)
