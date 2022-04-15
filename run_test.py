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
from environment.eva_policy import EVA_POLICY_L
from rl.multi_model_eva import MULTI_MODEL_EVA
import xlwt

############## Super Hyperparaters ####################

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
        self.eva_policy = EVA_POLICY_L()
        self.game_results = []
    
    def load_models(self):
        self.net_pur.load_models()
        self.net_eva.load_models()
    
    def test_one(self):
        win_times = 0
        for epo in range(100):
            state_temp = self.game.initial_environment(epo)
            results = 'NOT CATCH'
            for step in range(MAX_STEP):
                # action_pur = self.net_pur.get_action(state_temp, evalue= False)
                action_pur = self.net_pur.get_action(state_temp, evalue= True)

                # action_eva = self.net_eva.get_action(state_temp)
                action_eva = self.eva_policy.move_po(state_temp)
                # action_eva = self.eva_policy.move_random()
                next_state, rw_pur, rw_eva, done = self.game.step(action_pur, action_eva)
                state_temp = next_state
                if done == True:
                    results = 'CATCH'
                    win_times += 1
                    break
                # self.game.plot('map', False)
                # plt.pause(1e-2)
                # plt.cla()
            self.game.plot('map', True)
            # plt.pause(1)
            plt.cla()
        # plt.show()
            # print('test episode %d pur %s'%(epo, results))
        print('win rates is %d'%win_times)
        return win_times

    def test_two(self):
        pass

    def test_three(self):
        pass


if __name__ == "__main__":
    # book = xlwt.Workbook() 
    # sheet = book.add_sheet(u'sheet1',cell_overwrite_ok=True)
    # sheet.write(0,0,'proposed') 
    a = Game_Test()
    a.load_models()
    # results = []
    # for i in range(50):
    #     sheet.write(i+1,0,a.test_one())
    # book.save('Experiment_resluts.xls')
    a.test_one()

