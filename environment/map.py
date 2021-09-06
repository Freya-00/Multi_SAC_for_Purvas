
#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: map.py
# Creat Time: 2021-09-05 16:59:11 星期天
# Version: 1.0

# Description: for the map 

"""
Create the map for the pursuit evasion game
author: Jiebang
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

class PurEvaMap(object):
    '''
    Map:
                length
        ------------------------
        -                      - 
        -                      -
        -                      - width
        -                      -
        -                      -
       0------------------------
    '''
    def __init__(self, length, width):
        self.length = length # the length of map
        self.width = width # the width of map
        self.obs_density = 20
        self.obs_radius = 4
        self.obstacle = []
        for i in range(int(self.length/self.obs_density)):
            for j in range(int(self.width/self.obs_density)):
                self.obstacle.append([(i+1)*self.obs_density, (j+1)*self.obs_density])
        
    def collection_detection(self, pos): 
        flag = False
        if pos[0] < 0 or pos[0] > self.length or pos[1] < 0 or pos[1] > self.width:
            flag = True
        elif self.obstacle_collision_detection(pos) == True:
            flag = True
        return flag

    def obstacle_collision_detection(self, pos):
        collision_flag = False
        def _cal_distance(a_pos,b_pos):
            return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
        
        for i in range(len(self.obstacle)):
            if _cal_distance(self.obstacle[i], pos) <= self.obs_radius:
                collision_flag = True
                break
        return collision_flag

    def plot_map(self):
        circle_theta = np.linspace(0, 2 * np.pi, 200)
        for i in range(len(self.obstacle)):
            circle_x = self.obs_radius*np.cos(circle_theta) + self.obstacle[i][0]
            circle_y = self.obs_radius*np.sin(circle_theta) + self.obstacle[i][1]
            plt.plot(circle_x,circle_y,color="darkred", linewidth=2)


if __name__ == "__main__":
    x = random.random()*100
    y = random.random()*100
    map = PurEvaMap(100,100)
    print(map.collection_detection([x,y]))
    map.plot_map()
    plt.scatter(x,y)
    plt.show()
