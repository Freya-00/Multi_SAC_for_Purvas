#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: uavplane.py
# Creat Time: 2021-06-01 20:19:59 星期二
# Version: 1.0

# Description: dynamic mode
# Last Modified time: 2021-07-05 17:06:19 星期一

VEL_DISCOUNT = 0.1
from math import pi, sin, cos
import math 


class UAVPLANE(object):

    def __init__(self,
                initial_pos,
                vel,
                initial_theta = 0.0,
                turning_angle = 12):
        
        self.vel = vel
        self.turning_angle = turning_angle
        self.initial_pos = initial_pos
        self.initial_theta = initial_theta
        self.x = [initial_pos[0]]
        self.y = [initial_pos[1]]
        self.theta = [initial_theta]

    def move(self, theta_d1, cosllision = False):
        """
        Input: theta_d1 between +-pi/12
        """
        theta_next = self.theta[-1] + theta_d1.tolist()[0]* pi/self.turning_angle
        if theta_next > pi:
            theta_next -= pi*2
        if theta_next < -pi:
            theta_next += pi*2
        
        self.theta.append(theta_next)
        if cosllision == True: 
            self.x.append(self.x[-1])
            self.y.append(self.y[-1])
            # self.x.append(self.x[-1] + self.vel* VEL_DISCOUNT* cos(self.theta[-1]))
            # self.y.append(self.y[-1] + self.vel* VEL_DISCOUNT* sin(self.theta[-1]))
        else:
            self.x.append(self.x[-1] + self.vel* cos(self.theta[-1]))
            self.y.append(self.y[-1] + self.vel* sin(self.theta[-1]))

    def get_next_pos(self, theta_d1):
        theta_next = self.theta[-1] + theta_d1.tolist()[0]* pi/self.turning_angle
        next_x = self.x[-1] + self.vel* cos(theta_next)
        next_y = self.y[-1] + self.vel* sin(theta_next)
        return [next_x, next_y]

    def reset_theta(self, taregt_pos):
        self.initial_theta = math.atan2(taregt_pos[1] - self.initial_pos[1], taregt_pos[0] - self.initial_pos[0])
    
    def reset_dynamic_model(self):
        self.x = [self.initial_pos[0]]
        self.y = [self.initial_pos[1]]
        self.theta = [self.initial_theta]
    
    def get_pos(self):
        return [self.x[-1], self.y[-1]]
    

if __name__ == "__main__":
    pass

