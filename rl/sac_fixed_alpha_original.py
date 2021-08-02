#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: sac.py
# Creat Time: 2021-06-01 20:20:25 星期二
# Version: 1.0

# Description: SAC networks

import os

from torch._C import set_flush_denormal

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import sys

sys.path.append("../code")
import math
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, normal

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action


def plot(frame_idx, rewards):
    plt.figure()
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # normal = Normal(0, 1)
        # z      = normal.sample()* std + mean 
        normal = Normal(mean,std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        log_prob = Normal(mean, std).log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # normal = Normal(0, 1)
        # z      = normal.sample()* std + mean 
        normal = Normal(mean,std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        action  = action.detach().cpu().numpy()
        return action[0]


class SACnetworks(object):
    def __init__(self,
                label,
                state_dim,
                action_dim,
                hidden_dim = 256,
                batch_size = 128,
                value_lr = 3e-4,
                soft_q_lr = 3e-4,
                policy_lr = 3e-3,
                replay_buffer_size = 1000000
                ):
        self.label = label
        self.hidden_dim = hidden_dim # the nodes of hidden layer
        self.batch_size = batch_size # batch size of memory pool
        self.value_lr = value_lr # learning rate of value 
        self.soft_q_lr = soft_q_lr # learning rate of q 
        self.policy_lr = policy_lr # learning rate of policy 
        self.replay_buffer_size = replay_buffer_size

        self.value_net =        ValueNetwork(state_dim, hidden_dim).to(device)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

        self.soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        
        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        self.value_optimizer  = optim.Adam(self.value_net.parameters(), 
                                            lr=self.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), 
                                            lr=self.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), 
                                            lr=self.policy_lr)

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def get_action(self,state):
        return self.policy_net.get_action(state)

    def push_data_to_buffer(self,state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def networks_update(self,
                    gamma=0.99,
                    mean_lambda=1e-3,
                    std_lambda=1e-3,
                    z_lambda=0.0,
                    soft_tau=1e-2,
          ):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        expected_q_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)


        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean() # 没有看懂 polcy的更新还存在一些问题
        
        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss  = std_lambda  * log_std.pow(2).mean()
        z_loss    = z_lambda    * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        
        
        
    def save_model(self):
        torch.save(self.value_net.state_dict(),
                        'rl/save_model/%s_value_net.pth'%self.label)
        torch.save(self.target_value_net.state_dict(),
                        'rl/save_model/%s_target_value_net.pth'%self.label)
        torch.save(self.soft_q_net.state_dict(),
                        'rl/save_model/%s_soft_q_net.pth'%self.label)
        torch.save(self.policy_net.state_dict(),
                        'rl/save_model/%s_policy_net.pth'%self.label)
        
        torch.save(self.value_optimizer.state_dict(), 
                        'rl/save_model/%s_value_optimizer.pth'%self.label)
        torch.save(self.soft_q_optimizer.state_dict(), 
                        'rl/save_model/%s_soft_q_optimizer.pth'%self.label)
        torch.save(self.policy_optimizer.state_dict(), 
                        'rl/save_model/%s_policy_optimizer.pth'%self.label)

    
    def load_model(self):
        self.value_net.load_state_dict(
                torch.load('rl/save_model/%s_value_net.pth'%self.label))
        self.target_value_net.load_state_dict(
                torch.load('rl/save_model/%s_target_value_net.pth'%self.label))
        self.soft_q_net.load_state_dict(
                torch.load('rl/save_model/%s_soft_q_net.pth'%self.label))
        self.policy_net.load_state_dict(
                torch.load('rl/save_model/%s_policy_net.pth'%self.label))
        
        self.value_optimizer.load_state_dict(
                torch.load('rl/save_model/%s_value_optimizer.pth'%self.label))
        self.soft_q_optimizer.load_state_dict(
                torch.load('rl/save_model/%s_soft_q_optimizer.pth'%self.label))
        self.policy_optimizer.load_state_dict(
                torch.load('rl/save_model/%s_policy_optimizer.pth'%self.label))





if __name__ == "__main__":
    max_frames  = 40000
    max_steps   = 500
    frame_idx   = 0
    rewards     = []
    

    env = NormalizedActions(gym.make("Pendulum-v0"))
    # env = gym.make("Pendulum-v0")
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    net = SACnetworks('pur1',state_dim,action_dim)
    # net.load_model()

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = net.get_action(state)
            next_state, reward, done, _ = env.step(action)

            net.push_data_to_buffer(state, action, reward, next_state, done)
            if len(net.replay_buffer) > net.batch_size:
                net.networks_update()
            
            state = next_state
            episode_reward += reward
            frame_idx += 1
            
            if frame_idx % 10000 == 0:
                net.save_model()
                plot(frame_idx, rewards)
            
            if done:
                net.save_model()
                break
        print('reward' )
        rewards.append(episode_reward)
