#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: eva_potential.py
# Creat Time: 2021-11-14 21:26:26 星期天
# Version: 1.0

# Description: 
import sys
sys.path.append("../code")
import math
import random
import numpy as np
from environment.map import PurEvaMap

OBS_POWER = 50
TURNIN_ANGEL = 12

class EVA_POLICY(object):
    def __init__(self):
        self.map = PurEvaMap()

    def move_random(self):
        ac = 1 if random.random() > 0.5 else -1
        ac = random.random()* ac
        return [ac]

    def move_po(self, state):
        'pos_pur: the position of pursuit'
        'pos_eva: pos and theta'
        pos_pur = state[-1][-8:]
        pos_pur = [i*10 for i in pos_pur]
        pos_eva = state[-1][:3]
        pos_eva[0] *= 10
        pos_eva[1] *= 10
        evade_vector = [0,0]
        min_obs_1 = self.map.get_min_n_obs([pos_eva[0], pos_eva[1]])
        for i in range(4):
            evade_vector[0] -= pos_pur[i*2+0]
            evade_vector[1] -= pos_pur[i*2+1]
        evade_vector[0] += pos_eva[0] * len(pos_pur)
        evade_vector[1] += pos_eva[1] * len(pos_pur)
        evade_vector_length = math.sqrt(evade_vector[0]**2 + evade_vector[1]**2)
        evade_vector[0] /= evade_vector_length
        evade_vector[1] /= evade_vector_length


        vector_1 = [pos_eva[0] - min_obs_1[0], pos_eva[1] - min_obs_1[1]]
        vector_1_length = math.sqrt(vector_1[0]**2 + vector_1[1]**2)
        if vector_1_length <= 12:
            obs_vector = [
                vector_1[0]*OBS_POWER/abs(vector_1_length - self.map.obs_radius), 
                vector_1[1]*OBS_POWER/abs(vector_1_length - self.map.obs_radius)
        ]
            vector_final = [
            evade_vector[0] + obs_vector[0],
            evade_vector[1] + obs_vector[1],
        ]
        else:
            vector_final = evade_vector
        

        ideal_angle = math.atan2(vector_final[1], vector_final[0])
        # print('ideal_angle',ideal_angle)
        ac = ideal_angle - pos_eva[2]
        # print('ac',ac)
        ac /= math.pi/12
        ac = np.array([ac])
        ac = np.clip(ac, -1.00, 1.00)

        return [ac]

    def repul(self):

        pass



        
