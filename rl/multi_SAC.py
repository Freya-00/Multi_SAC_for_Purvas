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
                                share_action= False,
                                )
            )

    def get_action(self, state, evalue = False):
        action = []
        for net in self.nets:
            action.append(net.get_action(state))
        return action
    
    def update_policy(self, state, action, reward, next_state, done):
        '需要对数据进行处理'
        for i in range(self.num_net):
            self.nets[i].memory.push(state, action[i], reward[i], next_state, done)
            self.nets[i].update_policy()

    def save_models(self):
        for net in self.nets:
            net.save_model('multi_pur_eva','%s_%s'%(net.label, time.time()))

    def load_models(self):
        for net in self.nets:
            net.load_model('rl/save_model/sac_actor_multi_pur_eva_%s'%net.label,
                        'rl/save_model/sac_critic_multi_pur_eva_%s'%net.label)



if __name__ == "__main__":
    pass

