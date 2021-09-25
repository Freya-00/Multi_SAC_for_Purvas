#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: run_stage_two.py
# Creat Time: 2021-09-17 09:51:26 星期五
# Version: 1.0

# Description: training for stage TWO
import sys

sys.path.append("../code")

from rl.multi_SAC import MULTI_SAC_NETWORKS
from environment.swamp_hunt import SWAMP_HUNT_GAME
import matplotlib.pyplot as plt

############## Super Hyperparaters ####################
EPOSIDE = 10000
MAX_STEP = 80
AC_DIM = 1
STATE_DIM = 15
PUR_NUM = 4

############## Main Class ############################
class Stage_Two(object):
    'pur catch not move target'
    def __init__(self):
        self.game = SWAMP_HUNT_GAME('two')
        self.net_pur = MULTI_SAC_NETWORKS('pur', PUR_NUM, AC_DIM, STATE_DIM)
        self.net_eva = MULTI_SAC_NETWORKS('eva', 1, 1, 15, flag_policy_deterministic= True)
        self.game_results = []
    
    def load_models(self):
        self.net_pur.load_models()
        self.net_eva.load_models()
    
    def run(self):
        for epo in range(EPOSIDE):
            state_temp = self.game.initial_environment(epo)
            results = 'NOT CATCH'
            for step in range(MAX_STEP):
                action_pur = self.net_pur.get_action(state_temp)
                action_eva = self.net_eva.get_action(state_temp)
                next_state, rw_pur, rw_eva, done = self.game.step(action_pur, action_eva)
                self.net_pur.update_policy(state_temp, action_pur, rw_pur, next_state, done)
                self.net_eva.update_policy(state_temp, action_eva, rw_eva, next_state, done)
                state_temp = next_state
                if done == True:
                    results = 'CATCH'
                    break
            if epo %2000 == 0 and epo > 0:
                self.game.plot('map', True)
                self.game.plot('reward', True)
                self.game.plot('time', True)
                # plt.show()
                pass
            if epo %2000 == 0 and epo >0:
                self.save_model()
                self.test_learn()
            print('episode %d pur %s'%(epo,results))

    def test_learn(self):
        win_times = 0
        for test_epo in range(100):
            state_temp = self.game.initial_environment(test_epo)
            results = 'NOT CATCH'
            for test_step in range(MAX_STEP):
                action_pur = self.net_pur.get_action(state_temp)
                action_eva = self.net_eva.get_action(state_temp)
                next_state, rw_pur, rw_eva, done = self.game.step(action_pur, action_eva)
                state_temp = next_state
                if done == True:
                    results = 'CATCH'
                    win_times += 1
                    break
            if test_epo % 10 ==0 and test_epo >0:
                self.game.plot('map', False)
                # self.game.plot('reward', False)
                # self.game.plot('time', False)
                plt.show()
            print('test episode %d pur %s'%(test_epo, results))
        print('win rates is %d'%win_times)
    
    def save_model(self):
        self.net_pur.save_models()
        self.net_eva.save_models()



if __name__ == "__main__":
    a = Stage_Two()
    a.load_models()
    # a.run()
    # a.save_model()


    a.test_learn()
