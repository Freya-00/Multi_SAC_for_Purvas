#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: swamp_hunt.py
# Creat Time: 2021-09-12 17:03:47 星期天
# Version: 1.0

# Description: the game of swamp hunt

from random import randrange
import sys

sys.path.append("../code")
import math
import numpy as np
from dynamic_model.uavplane import UAVPLANE
from environment.map import PurEvaMap
from environment.reward import PurEva_2D_Reward

"""
Definition of hyperparameters
"""
NUM_PUR = 4
NUM_EVA = 1
EVA_MOVE = False
CAPTURE_DISTANCE = 2 # Captured distance
VEL = 2
INITIAL_POS_PUR = [[7,7],[7,153],[93,7],[93,153]]
INITIAL_POS_EVA = [[70,50]]
"""
Core Class of game
"""
class SWAMP_HUNT_GAME(object):
    def __init__(self):
        self.pursuit = []
        self.evasion = []
        self.map = PurEvaMap()
        self.reward = PurEva_2D_Reward()
        self.dis = []
        for i in range(NUM_PUR):
            self.pursuit.append(UAVPLANE(INITIAL_POS_PUR[i], VEL))
            self.dis.append([])
        for j in range(NUM_EVA):
            self.evasion.append(UAVPLANE(INITIAL_POS_EVA[j], VEL))
        _ = self._update_env_parametres()
        

    def initial_environment(self):
        self._set_random_pos()
        self.dis = []
        for eva in self.evasion:
            eva.reset_dynamic_model()
        for pur in self.pursuit:
            pur.reset_dynamic_model()
            self.dis.append([])
        _ = self._update_env_parametres()
        return self._get_state()
    
    def step(self, action):
        'step the game'
        'action = [pur.ac, eva.ac]'
        ac_pur = action[0:-1]
        ac_eva = action[-1]
        for i in range(NUM_PUR):
            pos_tem = self.pursuit[i].get_next_pos(ac_pur[i])
            b_c, o_c = self.map.map_detect(pos_tem)
            self.pursuit[i].move(ac_pur[i], cosllision = (b_c or o_c))
        for j in range(NUM_EVA):
            pos_tem = self.evasion[j].get_next_pos(ac_eva[j])
            b_c, o_c = self.map.map_detect(pos_tem)
            self.evasion[j].move(ac_eva[j], cosllision = (b_c or o_c))
        game_done = self._update_env_parametres()
        pos_all = self._get_pos_all()
        rw_pur, rw_eva = self._return_reward(pos_all, self.dis, game_done)
        next_state = self._get_state()
        return next_state, rw_pur, rw_eva, game_done

    def _return_reward(self, pos_all, distance, game_done):
        return self.reward.return_reward(pos_all, distance, game_done)

    def _set_random_pos(self):
        for eva in self.evasion:
            eva.initial_pos = self.map.get_new_eva_pos()
        
        for pur in self.pursuit:
            pur.reset_theta(self.evasion.initial_pos)
        
    def _update_env_parametres(self):
        'including distance'
        game_done = False
        def _cal_distance(a_pos,b_pos):
            return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
        for i in range(NUM_PUR):
            self.dis[i].append(_cal_distance(self.pursuit[i].get_pos(),self.evasion[-1].get_pos()))
            if self.dis[i][-1] < CAPTURE_DISTANCE:
                game_done = True
        return game_done

    def _get_pos_all(self):
        pos_all = []
        for i in range(NUM_PUR):
            pos_all.append(self.pursuit[i].get_pos())
        for j in range(NUM_EVA):
            pos_all.append(self.evasion[j].get_pos())
        return pos_all
    
    def _get_state(self):
        state = []
        for i in range(NUM_PUR):
            state.append(self.pursuit[i].x[-1]/10)
            state.append(self.pursuit[i].y[-1]/10)
            state.append(self.pursuit[i].theta[-1])
        
        for j in range(NUM_EVA):
            state.append(self.evasion[j].x[-1]/10)
            state.append(self.evasion[j].y[-1]/10)
            state.append(self.evasion[j].theta[-1])
        
        return np.array(state)


    def plot(self):
        pass


