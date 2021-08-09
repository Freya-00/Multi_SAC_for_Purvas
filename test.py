#!/usr/bin/python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: test.py
# Creat Time: 2021-08-09 16:43:58 星期一
# Version: 1.0

# Description: test models

import sys

sys.path.append("../code")

from environment.pureva_2D import PurEva_2D_Game
import matplotlib.pyplot as plt



if __name__ == "__main__":
    num_pw = [0]
    EPOSIDES = 8000
    env = PurEva_2D_Game() # need load models
    final_test_win = 0
    for _ in range(50):
        env.set_random_position()
        env.initial_env()
        for j in range(env.max_step):
            _, done, results = env.act_and_learn(j, eval_f = True)
            if done == True:
                if results[0] == 'purs win':
                    final_test_win += 1
                break
    print(final_test_win)



