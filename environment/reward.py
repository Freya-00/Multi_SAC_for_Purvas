#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: reward.py
# Creat Time: 2021-06-24 16:08:13 星期四
# Version: 1.0

# Description: for Curriculum Reward 
# Last Modified time: 2021-06-30 16:19:28 星期三
# Last Modified time: 2021-07-02 15:05:06 星期五

import math
import numpy as np
from random import random
import matplotlib.pyplot as plt
from numpy.lib.function_base import copy

CAPTURE_DISTANCE = 3 # Captured distance
MAX_STEP = 150
LENGTH = 100
WIDTH = 100
POS_OBSTACLE = [[LENGTH/4, WIDTH/4],
                [3*LENGTH/4, WIDTH/4], 
                [LENGTH/4, 3*WIDTH/4], 
                [3*LENGTH/4, 3*WIDTH/4]]
RADIUS_OBSTACLE = [15,15,15,15]
MARGIN = 0.5

class PurEva_2D_Reward(object):
    def __init__(self,
                vel_p,
                vel_e,
                dis_cap,
                TIME,
                map_length,
                map_width):
        self.vel_p = vel_p
        self.para_a = vel_p/vel_e
        # self.para_a = 0.25
        self.DIS_CAP = dis_cap
        self.TIME = TIME
        self.map_length = map_length
        self.map_width = map_width
        self.game_winner = []
        self.win_rate = []

    def return_reward(self, position_last, position_all, t):
        'position_all = [[P1],[P2],[P3],[E1]]'
        done, reward_pur, reward_eva, results = self._commom_reward_share(position_last, position_all, t)
        return done, reward_pur, reward_eva, results

    def _commom_reward_share(self, position_last, position_all, time):
        pos_pur_last = position_last[0:3]
        pos_eva_last = position_last[-1]
        dis_purs_eva_last = []
        for i in range(3):
            dis_purs_eva_last.append(self._cal_distance(pos_pur_last[i],pos_eva_last))
        pos_pur = position_all[0:3]
        pos_eva = position_all[-1]
        dis_purs_eva = [] # [d_p1e,d_p2e,d_p3e]
        r_pur_team_shaping = []
        r_pur_ending = []
        r_pur_all = []
        done = False
        game_results = None
        'get distance'
        for i in range(3):
            dis_purs_eva.append(self._cal_distance(pos_pur[i],pos_eva))
        dis_purs_eva = np.array(dis_purs_eva)
        dis_min = dis_purs_eva.min() # min distance
        'get eva reward'
        r_eva = self._eva_reward(dis_min, pos_eva)
        'get pur end reward'
        if dis_min < CAPTURE_DISTANCE :
            done = True
            for i in range(3):
                r_pur_ending.append(10)
            self._winrate_update(1)
            game_results = ['purs win', sum(self.game_winner)]
        elif time == MAX_STEP -1:
            done = True
            for i in range(3):
                r_pur_ending.append(0)
            self._winrate_update(0)
            game_results = ['eva win', len(self.game_winner) - sum(self.game_winner)]
        r_team_all = - math.exp(dis_min/100) + 1
        
        'get pur shaping reward'
        for i in range(3):
            r_pur_team_shaping = r_team_all* ((np.sum(dis_purs_eva) - dis_purs_eva[i])/ np.sum(dis_purs_eva))
            r_collision = self._punish_against_the_wall(position_all[i])
            if done == True:
                r_collision += r_pur_ending[i]
            r_pur_all.append(r_pur_team_shaping + r_collision)
        
        return done, r_pur_all, r_eva, game_results

    def _eva_reward(self, dis_min, pos_eva):
        re_evas= []
        for _ in range(1):
            r = math.log(dis_min/10)
            re_evas.append(r + self._punish_against_the_wall(pos_eva))
        return re_evas

    def _punish_against_the_wall(self, pos):
        r = 0
        for i in range(len(POS_OBSTACLE)):
            if self._cal_distance(POS_OBSTACLE[i], pos) <= (RADIUS_OBSTACLE[i] + MARGIN):
                r = -1
        if pos[0] >= (self.map_length-MARGIN) or pos[0] <= MARGIN or pos[1] <= MARGIN or pos[1] >= (self.map_width-MARGIN):
            r = -1
        return r

    def _winrate_update(self, result):
        self.game_winner.append(result)
        if len(self.game_winner) >= 10:
            if len(self.game_winner) < 500 :
                self.win_rate.append(sum(self.game_winner)/len(self.game_winner))
            else:
                self.win_rate.append(sum(self.game_winner[-500:])/len(self.game_winner[-500:]))

    def _cal_distance(self,a_pos,b_pos):
        'input [x,y]'
        return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
    

if __name__ == "__main__":
    # re = PurEva_2D_Reward(3.0,2.0,3.0,0.5,100,100)
    # pos_p = []
    # for i in range(3):
    #     a = random()*100
    #     b = random()*100
    #     pos = [a,b]
    #     plt.scatter(pos[0],pos[1])
    #     pos_p.append(pos)
    # # pos_p.append([80.0,10.0])
    # # pos_p.append([10.0,10.0])
    # # pos_p.append([10.0,80.0])
    # a = random()*100
    # b = random()*50
    # pos = [40.0,48.0]
    # plt.scatter(pos[0],pos[1],color = 'red')
    # plt.xlim(0,100)
    # plt.ylim(0,100)
    # pos_p.append(pos)
    # print(re.return_reward(pos_p))
    # plt.show()
    pass

