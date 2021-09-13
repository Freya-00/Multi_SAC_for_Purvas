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

EPOSIDE = 10000
MAX_STEP = 150
AC_DIM = 1
STATE_DIM = 12
class Stage_One(object):
    def __init__(self) -> None:
        def __init__(self):
            self.game = SWAMP_HUNT_GAME()
            self.net_pur = MULTI_SAC_NETWORKS('pur', 4, 1, 12)

    def run(self):
        for epo in range(EPOSIDE):
            state_temp = self.game.initial_environment()
            for step in range(MAX_STEP):
                action = self.net_pur.get_action()
                next_state, rw_pur, rw_eva, done = self.game.step(action)
                self.net_pur.update_policy(state_temp, action, rw_pur, next_state, done)
                state_temp = next_state
                

