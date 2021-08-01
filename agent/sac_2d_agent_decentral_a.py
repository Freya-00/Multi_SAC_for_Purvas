#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: sac_2d_agent.py
# Creat Time: 2021-06-01 20:19:47 星期二
# Version: 1.0

# Description: # Agent class for env including rl networks and dynamic model


from pickle import FALSE
import sys
import argparse
sys.path.append("../code")

import math
from random import random

import numpy as np

from agent.dynamic_model.uavplane import UAVPLANE

import argparse
import datetime
import gym
import numpy as np
import itertools
import torch

from rl.sac_aversion.sac import SAC
from rl.sac_aversion.replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()



class PurEva_2D_Agent(object):
    def __init__(self,
                agent_label,
                state_dim,
                action_dim,
                initial_x,
                initial_y,
                agent_vel,
                creat_network = True,
                share_action = False,
                load_existing_model = False
                    ):
        self.label = agent_label # the name of agent
        self.network_exist = creat_network # whether to create a network
        self.share_action = share_action
        self.load_existing_model = load_existing_model # whether to load model existted
       
        self.dynamic_model = UAVPLANE(initial_x, initial_y, agent_vel)
        
        if self.network_exist == True:
            if self.share_action == True:
                self.net = SAC(state_dim, action_dim, args, share_action = True)
            else:
                self.net = SAC(state_dim, action_dim, args, share_action = False)
            self.memory = ReplayMemory(args.replay_size, args.seed)
            # if self.share_action == True:
            #     self.net = SAC(state_dim, action_dim, args)
            #     self.memory = ReplayMemory(args.replay_size, args.seed)
            # else:
            #     self.net = rl.sac.SACnetworks(self.label,state_dim,action_dim)
        
        if self.load_existing_model == True:
            self.net.load_model()

    def get_action(self, state, evalue = FALSE):
        if self.network_exist == False:
            'default action for test'
            return self.action_eva_simple(state)
        else:
            if evalue:
                return self.net.select_action(state, evaluate = True)
            else:
                return self.net.select_action(state)  # Sample action from policy
    
    def update_policy(self):
        if len(self.memory) > args.batch_size:
            _, _, _, _, _, = self.net.update_parameters(self.memory)
            # print('learn')

    def action_eva_simple(self,state):
        p1x = state[0]*10
        p1y = state[1]*10
        p1v = state[2]
        p2x = state[3]*10
        p2y = state[4]*10
        p2v = state[5]
        p3x = state[6]*10
        p3y = state[7]*10
        p3v = state[8]
        e1x = state[9]*10
        e1y = state[10]*10
        e1v = state[11]
        px = (p1x + p2x + p3x)/3
        py = (p1y + p2y + p3y)/3
        dis = []
        for i in range(3):
            dis.append(self._cal_dis(state[3*i:3*i+2],state[9:11])*10)
        # min_pur = list.index(min(dis))
        min_dis = min(dis)
        if min_dis < 4:
            ac = np.array([1.0])
        else:
            theta_ideal = math.atan2((e1y-py),(e1x-px))
            ac = theta_ideal - e1v
            ac = np.clip(ac/math.pi,-1/3,1/3)
            ac = np.array([ac])
        return ac

    def _cal_dis(self,a_pos,b_pos):
        'input [x,y]'
        return math.sqrt((a_pos[0]-b_pos[0])**2 +
                        (a_pos[1]-b_pos[1])**2)


if __name__ == "__main__":
    pass
