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
"""
Definition of hyperparameters
"""
MARGIN = 3
REWARD_COLLISION = -3
SHAPING_GAMMA = 0.9

"""
Core Class of game
"""
class PurEva_2D_Reward(object):
    def __init__(self,
                num_pur,
                num_eva,
                map_length,
                map_width,
                obs_pos,
                obs_radius
                    ):
        self.num_pur = num_pur
        self.num_eva = num_eva
        self.map_length = map_length
        self.map_width = map_width
        self.obs_pos = obs_pos
        self.obs_radius = obs_radius

    def return_reward(self, pos_all, distance, game_done):
        'position_all = [[P1],[P2],[P3],[E1]]'
        reward_pur = self._course_one_pur(pos_all, distance, game_done)
        reward_eva = self._eva_reward(pos_all, distance)
        return reward_pur, reward_eva

    def _course_one_pur(self, pos_all, distance, game_done):
        r = []
        for i in range(self.num_pur):
            r_collision = self._punish_against_the_wall(pos_all[i])
            r_shaping = (distance[i][-2] - distance[i][-1])* SHAPING_GAMMA
            r_done = 0
            r.append(r_collision + r_shaping + r_done)
        return r

    def _eva_reward(self, pos_all, distance):
        re_evas= []
        pos_eva = pos_all[-1]
        dis_last = []
        for i in range(self.num_pur):
            dis_last.append(distance[i][-1])
        dis_min = np.array(dis_last).min()
        for _ in range(1):
            r = math.log(dis_min/10)
            re_evas.append(r + self._punish_against_the_wall(pos_eva))
        return re_evas

    def _punish_against_the_wall(self, pos):
        r = 0
        for i in range(len(self.obs_pos)):
            if self._cal_distance(self.obs_pos[i], pos) <= (self.obs_radius + MARGIN):
                r = REWARD_COLLISION
                break
        if pos[0] >= (self.map_length-MARGIN) or pos[0] <= MARGIN or pos[1] <= MARGIN or pos[1] >= (self.map_width-MARGIN):
            r = REWARD_COLLISION
        return r

    def _cal_distance(self,a_pos,b_pos):
        'input [x,y]'
        return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
    

if __name__ == "__main__":
    pass

