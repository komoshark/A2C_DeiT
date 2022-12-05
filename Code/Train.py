# YOUR CODE HERE
# import tensorflow as tf
# import torch
import os
import time
import numpy as np
import pandas as pd
import torch
import gym
import sys
import torch.nn as nn
import torch.optim as optim
from Model import A2C_DeiT, Simple_A2C
import torchvision.transforms as T
import matplotlib.pyplot as plt

"""This script defines the training, validation and testing process.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_lengths = []
average_lengths = []
all_rewards = []
saved_per_frames = 1e6
sum_frames = 1e6

def train(args, env):
    action_num = env.action_space.n
    entropy_term = 0
    actor_critic = A2C_DeiT(args.historic_num, action_num)
    actor_critic = actor_critic.to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr= args.learning_rate)
    frames = 0
    start_time = time.time()
    saved_per_frames = 1e3
    for episode in range(args.max_episodes):
        log_probs = []
        values = []
        rewards = []
        advantages = []
        state = env.reset()
        actions = []
        diss = []
        # repeat n times
        # states = np.tile(state, (args.historic_num, 1, 1))
        state = env.reset()
        states = [state for i in range(args.historic_num)]
        for step in range(args.num_steps):
            frames += 1
            # forward
            value, dist = actor_critic.forward(states)
            value = value.detach().cpu().numpy()[0][0]
            dist = dist.detach().cpu().numpy()[0]
            #print('value', value)
            #print('dist', dist)
            # choose and take action
            action = np.random.choice(action_num, p=np.squeeze(dist))
            actions.append(action)
            log_prob = np.log(dist[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            next_state, reward, done, _ = env.step(action)
            # add variables to list
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            # set historic states
            state = next_state
            states.append(state)
            del states[0]
            # set advantage
            value, _ = actor_critic.forward(states)
            value = value.detach().cpu().numpy()
            advantage = reward + args.GAMMA * value
            advantages.append(advantage)
            # done or steps expired
            if done or step == args.num_steps - 1:
                store_epoch(args, rewards, step, episode, frames, start_time, actor_critic)
                break
        # update actor critic
        #print('actions', actions)
        #print('values', values)
        #rint('train start loss func')
        values = torch.from_numpy(np.array(values)).to(device)
        advantages = torch.from_numpy(np.array(advantages)).to(device)
        log_probs = torch.from_numpy(np.array(log_probs)).to(device)
        loss = ac_loss(log_probs, advantages, values, entropy_term)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def store_epoch(args, rewards, step, epoch, frames, start_time, model):
  global saved_per_frames, sum_frames
  if epoch >0 and epoch%100 == 0:
    duration = time.time() - start_time
    all_rewards.append(np.sum(rewards))
    all_lengths.append(step)
    average_lengths.append(np.mean(all_lengths[-10:]))
    s_Kframes = "After training {:.2f} mins, episode: {}, frames: {:.2f}K, average reward: {}".format(
        duration/60, epoch, frames/1e3, np.mean(all_rewards[-10:]))
    s_Mframes = "After training {:.2f} hours, episode: {}, frames: {:.2f}M, average reward: {}".format(
        duration/3600, epoch, frames/1e6, np.mean(all_rewards[-10:]))
    if frames < 1e6:
      printlog(args, s_Kframes)
    else:
      printlog(args, s_Mframes)
      if frames > sum_frames: 
        printlog(args, 'save model in model{:.0f}.tar'.format(frames/saved_per_frames))
        torch.save(model.state_dict(), args.save_dir+'model{:.0f}.tar'.format(frames/saved_per_frames))
        sum_frames += saved_per_frames

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()

def ac_loss(log_probs, advantages, values, entropy_term, requires_grad=True):
    deltas = advantages - values
    critic_loss = 0.5 * deltas.pow(2).sum().requires_grad_(True)
    #print('critic_loss',critic_loss.requires_grad)
    actor_loss = (-log_probs * deltas).sum().requires_grad_(True)
    ac_loss = actor_loss + critic_loss + 0.1 * entropy_term
    return ac_loss

def plot():
    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()



### END CODE HERE
