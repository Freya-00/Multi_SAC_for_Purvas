#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: run_stage_one.py
# Creat Time: 2021-09-13 10:53:37 星期一
# Version: 1.0

# Description: run for learning stage one

import sys

sys.path.append("../code")

from rl.multi_SAC import MULTI_SAC_NETWORKS
from environment.swamp_hunt import SWAMP_HUNT_GAME
import matplotlib.pyplot as plt
import numpy as np
import time
############## Super Hyperparaters ####################
EPOSIDE = 10000
MAX_STEP = 80
AC_DIM = 1
STATE_DIM = 8 # x, y, theta, px1, py1, tx, ty, d_obs
PUR_NUM = 4
############## Main Class ############################
class Stage_One(object):
    'pur catch not move target'
    def __init__(self):
        self.game = SWAMP_HUNT_GAME('one')
        self.net_pur = MULTI_SAC_NETWORKS('pur', PUR_NUM, AC_DIM, STATE_DIM, flag_critic_more_infor = False)
        self.game_results = []

    def load_models(self):
        self.net_pur.load_models()
    
    def run(self):
        for epo in range(EPOSIDE):
            state_temp = self.game.initial_environment(epo)
            results = 'NOT CATCH'
            for step in range(MAX_STEP):
                action_pur = self.net_pur.get_action(state_temp)
                action_eva = [np.array([0])]
                next_state, rw_pur, rw_eva, done = self.game.step(action_pur,action_eva)
                self.net_pur.update_policy(state_temp, action_pur, rw_pur, next_state, done)
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
                action_pur = self.net_pur.get_action(state_temp, evalue=True)
                action_eva = [np.array([0])]
                next_state, _, _, done = self.game.step(action_pur, action_eva)
                # self.net_pur.update_policy(state_temp, action, rw_pur, next_state, done)
                state_temp = next_state
                if done == True:
                    results = 'CATCH'
                    win_times += 1
                    break
            # if test_epo % 10 ==0 and test_epo >0:
            #     self.game.plot('map', False)
            #     # self.game.plot('reward', False)
            #     # self.game.plot('time', False)
            #     plt.show()
            print('test episode %d pur %s'%(test_epo, results))
        print('win rates is %d'%win_times)
    
    def save_date(self):
        np.savetxt("reward_%s.txt"%(time.strftime("%Y-%m-%d", time.localtime())),self.game.reward_record_all)
        np.savetxt("time_%s.txt"%(time.strftime("%Y-%m-%d", time.localtime())),self.game.game_time)
        np.savetxt("catch_%s.txt"%(time.strftime("%Y-%m-%d", time.localtime())),self.game.pur_success_time)


    def save_model(self):
        self.net_pur.save_models()

if __name__ == "__main__":
    a = Stage_One()
    a.run()
    a.save_date()
    a.save_model()

    # a.load_models()
    # a.test_learn()