
#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: map.py
# Creat Time: 2021-09-05 16:59:11 星期天
# Version: 1.0

# Description: for the map 

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import heapq

''' High Parameters'''
OBS_DENSITY = 25
OBS_RADIUS = 5
MARGIN = 0.1

'''Class Defination'''
class PurEvaMap(object):

    def __init__(self):
        self.length = 160 # the length of map
        self.width = 100 # the width of map
        self.obs_radius = OBS_RADIUS
        self.obstacle = []
        for i in range(int(self.length/OBS_DENSITY-1)):
            for j in range(int(self.width/OBS_DENSITY-1)):
                self.obstacle.append([(i+1)*OBS_DENSITY, (j+1)*OBS_DENSITY])
    
    def map_detect(self, pos):
        'return wether collison of map board and obs'
        return self._board_detect(pos), self._obs_detect(pos)
    
    def get_new_eva_pos(self):
        'get a new random pos for eva'
        x = random.random()*100+30
        y = random.random()*50+30
        # x = random.random()*60+30
        # y = random.random()*40+30
        # x = round(random.random()*60+30)
        # y = round(random.random()*40+30)
        while self._obs_detect([x,y]):
            # x = round(random.random()*60+30)
            # y = round(random.random()*40+30)
            # x = random.random()*60+30
            # y = random.random()*40+30
            x = random.random()*100+30
            y = random.random()*50+30
        return [x,y]

    def get_new_pur_pos(self):
        'get a new random pos for pur'
        pass

    def _board_detect(self, pos):
        'pos = [x,y]'
        flag = False
        if pos[0] < 0 or pos[0] > self.length or pos[1] < 0 or pos[1] > self.width:
            flag = True
        return flag

    def _obs_detect(self, pos):
        collision_flag = False
        def _cal_distance(a_pos,b_pos):
            return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
        
        for i in range(len(self.obstacle)):
            if _cal_distance(self.obstacle[i], pos) <= self.obs_radius:
                collision_flag = True
                break
        return collision_flag
    
    def get_mindis_obs(self, pos):
        def _cal_distance(a_pos,b_pos):
            return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
        
        dis_obs = []
        for i in range(len(self.obstacle)):
            dis_obs.append(_cal_distance(self.obstacle[i], pos))
        
        return min(dis_obs)

    def get_min_n_obs(self, pos, n = 1):
        def _cal_distance(a_pos,b_pos):
            return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
        dis_obs = []
        for i in range(len(self.obstacle)):
            dis_obs.append(_cal_distance(self.obstacle[i], pos))
        
        min_num_index_list = map(dis_obs.index, heapq.nsmallest(n, dis_obs))
        min_num_index_list = list(min_num_index_list)
        return self.obstacle[min_num_index_list[0]]

    def plot_map(self):
        map_board_x = np.arange(0,self.length)
        map_board_y = np.arange(0,self.width)
        plt.plot(map_board_x,0*map_board_x, ':', color='black')
        plt.plot(map_board_x,0*map_board_x + self.width, ':', color='black')
        plt.plot(0*map_board_y, map_board_y, ':', color='black')
        plt.plot(0*map_board_y + self.length,map_board_y, ':', color='black')
        circle_theta = np.linspace(0, 2 * np.pi, 200)
        for i in range(len(self.obstacle)):
            circle_x = self.obs_radius*np.cos(circle_theta) + self.obstacle[i][0]
            circle_y = self.obs_radius*np.sin(circle_theta) + self.obstacle[i][1]
            plt.plot(circle_x,circle_y,color="darkred", linewidth=2)
        plt.axis('scaled')
        

if __name__ == "__main__":
    x = random.random()*100
    y = random.random()*100
    map1 = PurEvaMap()
    print(map1.map_detect([x,y]))
    map1.plot_map()
    # plt.scatter(x,y)
    print(map1.get_min_n_obs([x,y]))
    plt.show()
