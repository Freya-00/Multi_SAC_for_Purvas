#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: multi_model_eva.py
# Creat Time: 2021-09-25 16:15:33 星期六
# Version: 1.0

# Description: 
import sys
sys.path.append("../code")
from rl.SAC_agent_single import PurEva_2D_Agent
import numpy as np
import random

class MULTI_MODEL_EVA(object):
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
        choosed = random.randrange(self.num_net)
        return [self.nets[choosed].get_action(state, evalue = evalue)]
    
    def update_policy(self, state, action, reward, next_state, done):
        '需要对数据进行处理'
        for i in range(self.num_net):
            state_single = state
            state_single_next = next_state
            self.nets[i].memory.push(state_single, action, reward, state_single_next, done)
            self.nets[i].update_policy()

    def save_models(self):
        for net in self.nets:
            net.save_models()

    def load_models(self):
        for net in self.nets:
            net.load_models()


if __name__ == "__main__":
    pass

