#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
from random import randint, random
import numpy as np
from numpy.testing._private.utils import print_assert_equal
class A(object):
    def __init__(self):
        self.a = 1

    def test1(self):
        def test2():
            print('test2')
        print('test1')
        test2()

def _get_state_purs(state_polocy, action_purs):
    'both agents have same state space'
    state_purs = []
    for i in range(3):
        state_purs.append([state_polocy])
        for j in range(3):
            if j != i:
                state_purs[i] = np.append(state_purs[i], action_purs[j])
    return state_purs

if __name__ == "__main__":
    color = ['blue','green','red']
    # a = np.array([0.0])
    # b = np.array([1.0])
    # c = np.append(a,b).tolist()[0]
    # print(c)
    # print(type(c))
    # a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # print(a[:-3])
    # a = A()
    # a.test1()
    a = np.array([1,2,3,4,5,6,7,8,9])
    b = np.array([4,5,6])
    # a = np.append(a,b)
    # print(a)
    # print(b)
    # print(c)
    # test 
    print(_get_state_purs(a,b))
