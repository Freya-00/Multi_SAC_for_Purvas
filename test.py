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
import numpy as np
import random

if __name__ == "__main__":
    a = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    for i in range(4):
        b = a.copy()
        for j in range(3):
            b[j] = a[i*3+j]
            b[3+j] = a[12+j]
            b[6+j] = a[3+j] if i == 0 else a[j]
            b[9+j] = a[6+j] if i == 0 or i == 1 else a[3+j]
            b[12+j] = a[9+j] if i != 3 else a[6+j]
        print(b)


