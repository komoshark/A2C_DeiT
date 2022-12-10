### YOUR CODE HERE
# import tensorflow as tf
# import torch
import argparse
import numpy as np
import gym
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from scipy.signal import lfilter
from Train import train
import torch.optim as optim
import torch
import os
import faulthandler; faulthandler.enable()

colab_drive_path = '/content/drive/MyDrive/636Homework/CSCE636-project-2022/saved_models/predictions.npy'
local_path = '../models/'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-f')
    parser.add_argument('--env', default='Breakout-v4', type=str, help='gym environment')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--GAMMA', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--max_episodes', default=100000, type=int, help='max episodes')
    parser.add_argument('--historic_num', default=3, type=int, help='how many past states used')
    parser.add_argument('--num_steps', default=300, type=int, help='maximum steps')
    parser.add_argument('--save_dir', default=local_path, type=str, help='maximum steps')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    env = gym.make("Breakout-v4")
    train(args, env)