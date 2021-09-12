#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: swamp_hunt.py
# Creat Time: 2021-09-12 17:03:47 星期天
# Version: 1.0

# Description: the game of swamp hunt

import sys
sys.path.append("../code")
from environment.reward import PurEva_2D_Reward
from environment.map import PurEvaMap

"""
Definition of hyperparameters
"""

CAPTURE_DISTANCE = 3 # Captured distance
MAX_STEP = 150

"""
Core Class of game
"""
class SWAMP_HUNT_GAME(object):
    def __init__(self):
        self.map = PurEvaMap()

    def step(self):
        pass

    def return_reward(self):
        pass

    def plot(self):
        pass


