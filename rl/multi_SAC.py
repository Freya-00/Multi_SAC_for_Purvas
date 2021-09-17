#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: multi_SAC.py
# Creat Time: 2021-09-11 15:34:11 星期六
# Version: 1.0

# Description: multi SAC
import sys
sys.path.append("../code")
import time
from rl.SAC_agent_single import PurEva_2D_Agent
import numpy as np

class MULTI_SAC_NETWORKS(object):
    def __init__(self,
                label,
                num_networks,
                dim_action,
                dim_state,
                flag_policy_deterministic = False,
                flag_creat_network = True,
                flag_share_action = False,
                flag_load_existing_model = False
                ):
        self.label = label
        self.num_net = num_networks
        self.nets = []
        for i in range(self.num_net):
            self.nets.append(
                PurEva_2D_Agent('%s_%d'%(self.label,i), dim_state, dim_action,
                                policy_deterministic= flag_policy_deterministic,
                                share_action= False,
                                flag_automatic_entropy_tuning = True
                                )
            )

    def get_action(self, state, evalue = False):
        action = []
        for i in range(self.num_net):
            if self.label == 'pur':
                st = np.append(state[i*3:i*3+2],state[-3:-1])
                print(st)
            action.append(self.nets[i].get_action(st, evalue = evalue))
        return action
    
    def update_policy(self, state, action, reward, next_state, done):
        '需要对数据进行处理'
        for i in range(self.num_net):
            if self.label == 'pur':
                state_single = np.append(state[i*3:i*3+2],state[-3:-1])
                state_single_next = np.append(next_state[i*3:i*3+2],state[-3:-1])
            self.nets[i].memory.push(state_single, action[i], reward[i], state_single_next, done)
            self.nets[i].update_policy()

    def save_models(self):
        for net in self.nets:
            net.save_models()

    def load_models(self):
        for net in self.nets:
            net.load_models()



if __name__ == "__main__":
    pass

