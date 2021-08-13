#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: sac_2d_agent.py
# Creat Time: 2021-06-01 20:19:47 星期二
# Version: 1.0

# Description: # Agent class for env including rl networks and dynamic model


import sys

sys.path.append("../code")
from random import random
import time
from rl.sac_adjust_alpha.replay_memory import ReplayMemory
from rl.sac_adjust_alpha.sac import SAC
from agent.dynamic_model.uavplane import UAVPLANE

REPLAY_BUFFER_SIZE = 1000000
SEED = 123456


class PurEva_2D_Agent(object):
    def __init__(self,
                agent_label,
                state_dim,
                action_dim,
                initial_x,
                initial_y,
                agent_vel,
                policy_deterministic = False,
                creat_network = True,
                share_action = False,
                load_existing_model = False
                    ):
        self.label = agent_label # the name of agent
        self.network_exist = creat_network # whether to create a network
        self.load_existing_model = load_existing_model # whether to load model existted
       
        self.dynamic_model = UAVPLANE(initial_x, initial_y, agent_vel)
        
        if self.network_exist == True:
            if share_action == True and policy_deterministic == True:
                self.net = SAC(state_dim, action_dim, share_action = True, policy_type = "Deterministic")
            elif share_action == True:
                self.net = SAC(state_dim, action_dim, share_action = True)
            else:
                self.net = SAC(state_dim, action_dim, share_action = False)
            self.memory = ReplayMemory(REPLAY_BUFFER_SIZE, SEED)
        
        if self.load_existing_model == True:
            self.load_models()

    def get_action(self, state, evalue = False):
        if self.network_exist == False:
            'default action for test'
            return self.action_eva_simple(state)
        else:
            if evalue:
                return self.net.select_action(state, evaluate = True)
            else:
                return self.net.select_action(state)  # Sample action from policy
    
    def update_policy(self):
        if len(self.memory) > 1024:
            _, _, _, _, _, = self.net.update_parameters(self.memory)
            # print('learn')

    def save_models(self):
        self.net.save_model('multi_pur_eva','%s_%s'%(self.label, time.time()))

    def load_models(self):
        self.net.load_model('rl/save_model/sac_actor_multi_pur_eva_%s'%self.label,
                        'rl/save_model/sac_critic_multi_pur_eva_%s'%self.label)
if __name__ == "__main__":
    pass
