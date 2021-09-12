#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: multi_SAC.py
# Creat Time: 2021-09-11 15:34:11 星期六
# Version: 1.0

# Description: multi SAC
import sys
sys.path.append("../code")

from rl.sac_adjust_alpha.replay_memory import ReplayMemory
from rl.sac_adjust_alpha.sac import SAC

class MULTI_SAC_NETWORKS(object):
    def __init__(self,
                num_networks,
                dim_action,
                dim_state,
                flag_policy_deterministic = False,
                flag_creat_network = True,
                flag_share_action = False,
                flag_load_existing_model = False
                ):
        self.num_net = num_networks
        self.nets = []
        if flag_policy_deterministic == True:
            policy_type = "Deterministic"
        else:
            policy_type = "Gaussian"
        
        if flag_creat_network == True:
            for i in range(num_networks):
                self.nets.append(SAC(dim_state, dim_action, policy_type = policy_type))



    def update_networks(self):
        pass
        
        
        
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

