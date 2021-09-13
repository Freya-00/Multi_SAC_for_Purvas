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

from torch._C import FloatStorageBase

sys.path.append("../code")

from environment.pureva_2D import PurEva_2D_Game
import matplotlib.pyplot as plt



if __name__ == "__main__":
    a = [1,2,3,45,67]
    b = False
    c = True
    d = (b or c)
    print(d)


