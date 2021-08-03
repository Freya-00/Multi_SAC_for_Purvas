#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: reward.py
# Creat Time: 2021-06-24 16:08:13 星期四
# Version: 1.0

# Description: for Curriculum Reward 
# Last Modified time: 2021-06-30 16:19:28 星期三
# Last Modified time: 2021-07-02 15:05:06 星期五

import math
import numpy as np
from random import random
import matplotlib.pyplot as plt
from numpy.lib.function_base import copy

CAPTURE_DISTANCE = 3 # Captured distance
MAX_STEP = 150

class PurEva_2D_Reward(object):
    def __init__(self,
                vel_p,
                vel_e,
                dis_cap,
                TIME,
                map_length,
                map_width):
        self.vel_p = vel_p
        self.para_a = vel_p/vel_e
        # self.para_a = 0.25
        self.DIS_CAP = dis_cap
        self.TIME = TIME
        self.map_length = map_length
        self.map_width = map_width
        self.dis_purs_self = []
        self.dis_purs_eva = []
        self.game_winner = []
        self.win_rate = []

    def return_reward(self,position_all, t):
        'position_all = [[P1],[P2],[P3],[E1]]'
        # done, reward = self._curriculum_reward(position_all)
        done, reward_pur, results = self._commom_reward_share(position_all, t)
        reward_eva = self._eva_reward(position_all)
        return done, reward_pur, reward_eva, results
        
    def _normal_reward(self,position,time):
        pos_pur = position[0:3]
        pos_eva = position[-1]
        dis_purs_eva = [] # [d_p1e,d_p2e,d_p3e]
        r_shaping = []
        r_team = []
        r_all = []
        done_game = 0
        done = False
        for i in range(3):
            dis_purs_eva.append(self._cal_distance(pos_pur[i],pos_eva))
            r_shaping.append(- math.log(dis_purs_eva[i]*3))
            if dis_purs_eva[i] <= CAPTURE_DISTANCE:
                done_game = 1 # purs win

        if done_game == 1:
            self.game_winner.append(1)
            if len(self.game_winner) < 500:
                self.win_rate.append(sum(self.game_winner)/len(self.game_winner))
            else:
                self.win_rate.append(sum(self.game_winner[-500:])/len(self.game_winner[-500:]))
            print('purs win',sum(self.game_winner))
            done = True
            for i in range(3):
                r_team.append((1 - dis_purs_eva[i])/(sum(dis_purs_eva))*10)
        elif time == MAX_STEP -1:
            done_game = 2 # eva win
            self.game_winner.append(0)
            if len(self.game_winner) < 500:
                self.win_rate.append(sum(self.game_winner)/len(self.game_winner))
            else:
                self.win_rate.append(sum(self.game_winner[-500:])/len(self.game_winner[-500:]))
            done = True
            for i in range(3):
                r_team.append(0)
            print('eva win',len(self.game_winner) - sum(self.game_winner))
        else:
            for i in range(3):
                r_team.append(0)
        
        for i in range(3):
            r_collision = self._punish_against_the_wall(position[i])
            r_all.append(r_shaping[i] + r_team[i] + r_collision)

        return done, r_all
    
    def _commom_reward_share(self, position_all, time):
        pos_pur = position_all[0:3]
        pos_eva = position_all[-1]
        dis_purs_eva = [] # [d_p1e,d_p2e,d_p3e]
        r_team_iso = []
        r_all = []
        done = False
        game_results = None
        for i in range(3):
            dis_purs_eva.append(self._cal_distance(pos_pur[i],pos_eva))
        dis_purs_eva = np.array(dis_purs_eva)
        dis_min = dis_purs_eva.min() # min distance
        
        if dis_min < CAPTURE_DISTANCE :
            done = True
            self.game_winner.append(1)
            if len(self.game_winner) < 500:
                self.win_rate.append(sum(self.game_winner)/len(self.game_winner))
            else:
                self.win_rate.append(sum(self.game_winner[-500:])/len(self.game_winner[-500:]))
            game_results = ['purs win',sum(self.game_winner)]
            # print('purs win',sum(self.game_winner))
        if time == MAX_STEP -1:
            done = True
            self.game_winner.append(0)
            if len(self.game_winner) < 500:
                self.win_rate.append(sum(self.game_winner)/len(self.game_winner))
            else:
                self.win_rate.append(sum(self.game_winner[-500:])/len(self.game_winner[-500:]))
            game_results = ['eva win',len(self.game_winner) - sum(self.game_winner)]
            # print('eva win',len(self.game_winner) - sum(self.game_winner))
        r_team_all = - math.exp(dis_min/80) + 1
        
        for i in range(3):
            r_team_iso = r_team_all* ((np.sum(dis_purs_eva) - dis_purs_eva[i])/ np.sum(dis_purs_eva))
            r_collision = self._punish_against_the_wall(position_all[i])
            r_all.append(r_team_iso + r_collision)
        
        return done, r_all, game_results
        

    def _eva_reward(self,position_all):
        pos_pur = position_all[0:3]
        pos_eva = position_all[-1]
        re_evas= []
        dis_recent = []
        for i in range(3):
            dis_recent.append(self._cal_distance(pos_pur[i],pos_eva))
        dis_recent = np.array(dis_recent)
        dis_min = dis_recent.min() # min distance
        for _ in range(1):
            r = math.log(dis_min/10)
            re_evas.append(r + self._punish_against_the_wall(pos_eva))
       
        return re_evas

    def _curriculum_reward(self,position):
        pos_pur = position[0:3]
        pos_eva = position[-1]
        self.dis_purs_self = [] # [d_p12,d_p23,d_31]
        self.dis_purs_eva = [] # [d_p1e,d_p2e,d_p3e]
        for i in range(3):
            j = i+1 if i < 2 else 0
            self.dis_purs_self.append(self._cal_distance(pos_pur[i],pos_pur[j]))
            self.dis_purs_eva.append(self._cal_distance(pos_pur[i],pos_eva))

        proportion_purs = [] # [s_p12e,s_p23e,s_p31e]
        for i in range(3):
            j = i+1 if i < 2 else 0
            proportion_purs.append(self._cal_triangle_area(self.dis_purs_eva[i],
                                                        self.dis_purs_eva[j],
                                                        self.dis_purs_self[i]))
        s_p123 = self._cal_triangle_area(self.dis_purs_self[0],self.dis_purs_self[1],self.dis_purs_self[2])
        
        # if sum(proportion_purs) <= (s_p123 + 5):
        #     reward = self._besiege_curriculum(position_all)
        # else:
        #     # print('out triangle')
        #     # reward = self._out_triangle(s_p123 - sum(proportion_purs))
        reward = self._dis_reward()
            # reward = self._wall_siegr_curriculum(pos_pur,pos_eva,s_p123)
        
        return reward

    def _besiege_curriculum(self,position_all):
        'triangel siege reward '
        reward = []

        for i in range(3):
            j = i+1 if i < 2 else 0
            a = i-1 if i > 0 else 2
            r_collision = self._punish_against_the_wall(position_all[i])
            if self.dis_purs_eva[i] <= (self.vel_p * self.TIME + self.DIS_CAP)*2:
                reward.append(- self.dis_purs_eva[i] + r_collision) # 0~-4.5
            elif self.dis_purs_self[i] <= self.para_a*(self.dis_purs_eva[i] + self.dis_purs_eva[j]):
            #     self.dis_purs_self[a] <= self.para_a*(self.dis_purs_eva[i] + self.dis_purs_eva[a]):
                reward.append(- math.log(self.dis_purs_eva[i]) + r_collision) # 0~-4.9
            else:
                # reward.append(- self.dis_purs[i] - self.dis_purs[j] + 0.5* self.dis_purs_eva[i] + r_collision)
                # reward.append(- (self.dis_purs_self[i] - self.para_a*(self.dis_purs_eva[i] + self.dis_purs_eva[j]))/20 -
                #                 (self.dis_purs_self[a] - self.para_a*(self.dis_purs_eva[i] + self.dis_purs_eva[a]))/20 +
                #                 r_collision)
                reward.append(-(self.dis_purs_self[i] - self.para_a*(self.dis_purs_eva[i] + self.dis_purs_eva[j]))/10 + r_collision)
                if reward[-1] > 0:
                    print('正了')
        return reward

    def _wall_siegr_curriculum(self,pos_pur,pos_eva,s_p123):
        'wall siege reward '
        pos_pur = np.array(pos_pur)
        dis_wall = [pos_eva[0],
                    self.map_width - pos_eva[1],
                    self.map_length - pos_eva[0],
                    pos_eva[0]]
        choosed_wall = dis_wall.index(min(dis_wall))
        if choosed_wall == 0 or choosed_wall == 2:
            sort = pos_pur[:,1].argsort() # from min to max according y
        else:
            sort = pos_pur[:,0].argsort() # from min to max according x
        pos_p1 = copy(pos_pur[sort[0]])
        pos_p2 = copy(pos_pur[sort[1]])
        pos_p3 = copy(pos_pur[sort[2]])
        pos_m1 = copy(pos_pur[sort[0]])
        pos_m2 = copy(pos_pur[sort[2]])
        if choosed_wall == 0:          # x = 0
            pos_m1[0] = 0.0
            pos_m2[0] = 0.0
        elif choosed_wall == 2:        # y = map.length
            pos_m1[0] = self.map_length
            pos_m2[0] = self.map_length
        elif choosed_wall == 1:        # y = map.width
            pos_m1[1] = self.map_width
            pos_m2[1] = self.map_width
        else:                          # y = 0
            pos_m1[1] = 0.0
            pos_m2[1] = 0.0
        pos_purswall_sorted = [pos_p1,pos_p2,pos_p3,pos_m2,pos_m1]
        
        dis_purswall = [] # [d_p1p2,d_p2p3,d_p3m2,d_m2m1,d_m1p1]
        dis_purswall_eva = [] # [d_p1e,dp2e,d_p3e,dm2e,d_m1e]
        for i in range(5):
            j = i+1 if i < 4 else 0
            dis_purswall.append(
                self._cal_distance(pos_purswall_sorted[i],pos_purswall_sorted[j]))
            dis_purswall_eva.append(
                self._cal_distance(pos_purswall_sorted[i],pos_eva))
        
        proportion_purs_walls = [] # [s_p1p2e,s_p2p3e,s_p3m2e,s_m2m1e,s_m1p1e]
        for i in range(5):
            j = i+1 if i < 4 else 0
            proportion_purs_walls.append(
                self._cal_triangle_area(dis_purswall_eva[i],
                                        dis_purswall_eva[j],
                                        dis_purswall[i]))
        s_p123m12 = self._cal_pentagon_area(dis_purswall[3], # d_m1m2
                                        dis_purswall[4], # d_p1m1
                                        dis_purswall[2], # d_p3m2
                                        s_p123)
        
        reward_before_sort = []
        for i in range(3):
            r_collision = self._punish_against_the_wall(pos_purswall_sorted[i])
            if sum(proportion_purs_walls) > s_p123m12:
                'outside of pentagon'
                reward_before_sort.append((s_p123m12 - sum(proportion_purs_walls))/1000 - math.log(dis_purswall_eva[i]) + r_collision)
            elif dis_purswall_eva[i] <= (self.vel_p* self.TIME + self.DIS_CAP)* 2:
                reward_before_sort.append(- dis_purswall_eva[i] + r_collision)
            elif dis_purswall[i] <= self.para_a*(dis_purswall_eva[i] + dis_purswall_eva[j]):
                reward_before_sort.append(- math.log(dis_purswall_eva[i]) + r_collision)
            else:
                j = i+1 if i < 4 else 0
                # # reward_before_sort.append(
                # #     -(self.dis_purs[i] - self.para_a*(self.dis_purs_eva[i] + self.dis_purs_eva[j])) + r_collision)
                if i == 0:
                #     reward_before_sort.append(- proportion_purs_walls[4]/500 - math.log(self.dis_purs_eva[i]) + r_collision) # s_p1m1e
                #     reward_before_sort.append(
                #         - math.log(dis_purswall[-1] + dis_purswall_eva[0]) + r_collision) # d_m1p1 + d_p1e
                    reward_before_sort.append(
                        - (dis_purswall[i] - self.para_a*(dis_purswall_eva[i] + dis_purswall_eva[j]))*0.5  - 
                            (dis_purswall[4] - self.para_a*(dis_purswall_eva[4] + dis_purswall_eva[0]))*0.5 + 
                            r_collision
                    )
                elif i == 1:
                #     reward_before_sort.append(- s_p123/500 - math.log(self.dis_purs_eva[i]) + r_collision) # s_p1p2p3
                #     reward_before_sort.append(
                #         - math.log(dis_purswall[0] + dis_purswall[1] +  dis_purswall_eva[1]) + r_collision) # d_p1p2,d_p2p3,d_p2e
                    reward_before_sort.append(
                        - (dis_purswall[i] - self.para_a*(dis_purswall_eva[i] + dis_purswall_eva[j])) + 
                            r_collision
                    )
                else:
                #     # reward_before_sort.append(- proportion_purs_walls[2]/500 - math.log(self.dis_purs_eva[i]) + r_collision) # s_p3m2e
                #     reward_before_sort.append(
                #         - math.log(dis_purswall[2] + dis_purswall_eva[2]) + r_collision) # d_p3m2 + d_p3e
                    reward_before_sort.append(
                        - (dis_purswall[i] - self.para_a*(dis_purswall_eva[i] + dis_purswall_eva[j]))*0.5 -
                            (dis_purswall[i+1] - self.para_a*(dis_purswall_eva[i+1] + dis_purswall_eva[i+2]))*0.5 + 
                            r_collision
                    )

        sort_back = sort[:].argsort() # resort the sort
        reward_after_sort = [] # change sort
        for i in range(3):
            reward_after_sort.append(reward_before_sort[sort_back[i]])
        return reward_after_sort
    
    def _dis_reward(self):
        reward = []
        for i in range(3):
            # if self.dis_purs_eva[i] >= (self.vel_p * self.TIME + self.DIS_CAP)*2:
            reward.append(- math.log(self.dis_purs_eva[i]*3))
            # else:
            #     reward.append(- self.dis_purs_eva[i])
        return reward

    def _out_triangle(self, delta):
        reward = []
        for i in range(3):
            reward.append(delta)
        return reward

    def _punish_against_the_wall(self,pos):
        dis_wall = [pos[0],
                    self.map_width - pos[1],
                    self.map_length - pos[0],
                    pos[0]]
        min_wall = min(dis_wall)
        if min_wall < 1.5:
            return -10
        else:
            return 0

    def _cal_triangle_area(self,a,b,c):
        p = (a+b+c)/2
        if (p-a) < 0 or (p-b) < 0 or (p-c)<0:
            s = 0
        else:
            s = math.sqrt(p* (p-a)* (p-b)* (p-c))
        return s
    
    def _cal_pentagon_area(self,d_m1m2,d_p1m1,d_p3m2,s_p123):
        '''
        M1  .............    P1
            .            .
            w             .
            a              . P2
            l             .
            l            .
        M2  .............    P3
        '''
        return (d_p1m1 + d_p3m2)* d_m1m2/2 + s_p123
    
    def _cal_distance(self,a_pos,b_pos):
        'input [x,y]'
        return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
    

if __name__ == "__main__":
    # re = PurEva_2D_Reward(3.0,2.0,3.0,0.5,100,100)
    # pos_p = []
    # for i in range(3):
    #     a = random()*100
    #     b = random()*100
    #     pos = [a,b]
    #     plt.scatter(pos[0],pos[1])
    #     pos_p.append(pos)
    # # pos_p.append([80.0,10.0])
    # # pos_p.append([10.0,10.0])
    # # pos_p.append([10.0,80.0])
    # a = random()*100
    # b = random()*50
    # pos = [40.0,48.0]
    # plt.scatter(pos[0],pos[1],color = 'red')
    # plt.xlim(0,100)
    # plt.ylim(0,100)
    # pos_p.append(pos)
    # print(re.return_reward(pos_p))
    # plt.show()
    pass

