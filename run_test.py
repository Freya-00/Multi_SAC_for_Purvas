#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: run_test.py
# Creat Time: 2021-11-16 15:39:15 星期二
# Version: 1.0

# Description: test

import sys

sys.path.append("../code")

from rl.multi_SAC import MULTI_SAC_NETWORKS
from environment.swamp_hunt import SWAMP_HUNT_GAME
import matplotlib.pyplot as plt
from rl.multi_model_eva import MULTI_MODEL_EVA
import numpy as np
from environment.eva_policy import EVA_POLICY
############## Super Hyperparaters ####################
EPOSIDE = 10001
MAX_STEP = 80
AC_DIM = 1
STATE_DIM = 8
PUR_NUM = 4

############## Main Class ############################
class Game_Test(object):
    'pur catch not move target'
    def __init__(self):
        self.game = SWAMP_HUNT_GAME('test')
        self.net_pur = MULTI_SAC_NETWORKS('pur', PUR_NUM, AC_DIM, STATE_DIM)
        self.net_eva = MULTI_MODEL_EVA('eva', 3, 1, 11, flag_policy_deterministic= True)
        self.eva_policy = EVA_POLICY()
        self.game_results = []
    
    def load_models(self):
        self.net_pur.load_models()
        self.net_eva.load_models()
    
    def test_one(self):
        win_times = 0
        self.load_models()
        for epo in range(100):
            state_temp = self.game.initial_environment(epo)
            results = 'NOT CATCH'
            for step in range(MAX_STEP):
                action_pur = self.net_pur.get_action(state_temp)
                action_eva = self.net_eva.get_action(state_temp)
                # action_eva = self.eva_policy.move_po(state_temp)
                # action_eva = self.eva_policy.move_random()
                next_state, rw_pur, rw_eva, done = self.game.step(action_pur, action_eva)
                state_temp = next_state
                if done == True:
                    results = 'CATCH'
                    win_times += 1
                    break
            print('test episode %d pur %s'%(epo, results))
        print('win rates is %d'%win_times)
           


    def test_two(self):
        pass

    def test_three(self):
        pass

    

if __name__ == "__main__":
    a = Game_Test()
    a.test_one()

