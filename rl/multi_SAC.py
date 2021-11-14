#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: multi_SAC.py
# Creat Time: 2021-09-11 15:34:11 星期六
# Version: 1.0

# Description: multi SAC
import sys
sys.path.append("../code")
from rl.SAC_agent_single import PurEva_2D_Agent
import numpy as np


class MULTI_SAC_NETWORKS(object):
    def __init__(self,
                label,
                num_networks,
                dim_action,
                dim_state,
                flag_policy_deterministic = False,
                flag_critic_more_infor = False,
                ):
        self.label = label
        self.num_net = num_networks 
        self.nets = []
        self.critic_more_infor = flag_critic_more_infor
        for i in range(self.num_net):
            self.nets.append(
                PurEva_2D_Agent('%s_%d'%(self.label,i), dim_state, dim_action,
                                policy_deterministic= flag_policy_deterministic,
                                share_action= flag_critic_more_infor,
                                flag_automatic_entropy_tuning = True
                                )
            )

    def get_action(self, state, evalue = False):
        action = []
        for i in range(self.num_net):
            st = state[i]
            action.append(self.nets[i].get_action(st, evalue = evalue))
        
        return action
    
    def update_policy(self, state, action, reward, next_state, done):
        '需要对数据进行处理'
        for i in range(self.num_net):
            state_single = state[i]
            state_single_next = next_state[i]
            self.nets[i].memory.push(state_single, action[i], reward[i], state_single_next, done)
            self.nets[i].update_policy()

    def save_models(self):
        for net in self.nets:
            net.save_models()

    def load_models(self):
        for net in self.nets:
            net.load_models()

    def change_rank_for_critic(self, state, num):
        st = state.copy()
        for j in range(3):
            st[j] = state[num*3+j]
            st[3+j] = state[12+j]
            st[6+j] = state[3+j] if num == 0 else state[j]
            st[9+j] = state[6+j] if num == 0 or num == 1 else state[3+j]
            st[12+j] = state[9+j] if num != 3 else state[6+j]
        return st

if __name__ == "__main__":
    pass

