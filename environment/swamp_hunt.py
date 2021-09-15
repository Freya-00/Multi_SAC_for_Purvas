#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: swamp_hunt.py
# Creat Time: 2021-09-12 17:03:47 星期天
# Version: 1.0

# Description: the game of swamp hunt

from random import randrange
import sys

sys.path.append("../code")
import math
import numpy as np
from environment.dynamic_model.uavplane import UAVPLANE
from environment.map import PurEvaMap
from environment.reward import PurEva_2D_Reward
import matplotlib.pyplot as plt
import time
"""
Definition of hyperparameters
"""
NUM_PUR = 4
NUM_EVA = 1
EVA_MOVE = False
CAPTURE_DISTANCE = 2 # Captured distance
VEL = 2
INITIAL_POS_PUR = [[7,7],[153,7],[153,93],[7,93]]
INITIAL_POS_EVA = [[70,50]]
"""
Core Class of game
"""
class SWAMP_HUNT_GAME(object):
    def __init__(self):
        self.pursuit = []
        self.evasion = []
        self.map = PurEvaMap()
        self.reward = PurEva_2D_Reward(4, 1, self.map.length, self.map.width, self.map.obstacle, self.map.obs_radius)
        self.game_time = []
        self.pur_success_time = []
        self.reward_record_all = []
        self.reward_record_once = []
        self.dis = []
        for i in range(NUM_PUR):
            self.pursuit.append(UAVPLANE(INITIAL_POS_PUR[i], VEL))
            self.dis.append([])
            self.reward_record_all.append([])
            self.reward_record_once.append(0)
        for j in range(NUM_EVA):
            self.evasion.append(UAVPLANE(INITIAL_POS_EVA[j], 0))
        _ = self._update_env_parametres()
        

    def initial_environment(self, epo):
        if epo != 0:
            for i in range(NUM_PUR):
                self.reward_record_all[i].append(self.reward_record_once[i]/self.game_time[-1])
        self.game_time.append(0)
        self._set_random_pos()
        self.dis = []
        self.reward_record_once = []
        for eva in self.evasion:
            eva.reset_dynamic_model()
        for pur in self.pursuit:
            pur.reset_dynamic_model()
            self.dis.append([])
            self.reward_record_once.append(0)
        _ = self._update_env_parametres()
        return self._get_state()
    
    def step(self, action):
        'step the game'
        'action = [pur.ac, eva.ac]'
        self.game_time[-1] += 1
        ac_pur = action[:]
        ac_eva = [np.array([0])]
        for i in range(NUM_PUR):
            pos_tem = self.pursuit[i].get_next_pos(ac_pur[i])
            b_c, o_c = self.map.map_detect(pos_tem)
            self.pursuit[i].move(ac_pur[i], cosllision = (b_c or o_c))
        for j in range(NUM_EVA):
            pos_tem = self.evasion[j].get_next_pos(ac_eva[j])
            b_c, o_c = self.map.map_detect(pos_tem)
            self.evasion[j].move(ac_eva[j], cosllision = (b_c or o_c))
        game_done = self._update_env_parametres()
        pos_all = self._get_pos_all()
        rw_pur, rw_eva = self._return_reward(pos_all, self.dis, game_done)
        for i in range(NUM_PUR):
            self.reward_record_once[i] += rw_pur[i]
        next_state = self._get_state()
        if game_done ==True:
            self.pur_success_time.append(self.game_time[-1])
        return next_state, rw_pur, rw_eva, game_done

    def _return_reward(self, pos_all, distance, game_done):
        return self.reward.return_reward(pos_all, distance, game_done)

    def _set_random_pos(self):
        for eva in self.evasion:
            eva.initial_pos = self.map.get_new_eva_pos()
        
        for pur in self.pursuit:
            pur.reset_theta(self.evasion[-1].initial_pos)
        
    def _update_env_parametres(self):
        'including distance'
        game_done = False
        def _cal_distance(a_pos,b_pos):
            return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
        for i in range(NUM_PUR):
            self.dis[i].append(_cal_distance(self.pursuit[i].get_pos(),self.evasion[-1].get_pos()))
            if self.dis[i][-1] < CAPTURE_DISTANCE:
                game_done = True
        return game_done

    def _get_pos_all(self):
        pos_all = []
        for i in range(NUM_PUR):
            pos_all.append(self.pursuit[i].get_pos())
        for j in range(NUM_EVA):
            pos_all.append(self.evasion[j].get_pos())
        return pos_all
    
    def _get_state(self):
        state = []
        for i in range(NUM_PUR):
            state.append(self.pursuit[i].x[-1]/10)
            state.append(self.pursuit[i].y[-1]/10)
            state.append(self.pursuit[i].theta[-1])
        
        for j in range(NUM_EVA):
            state.append(self.evasion[j].x[-1]/10)
            state.append(self.evasion[j].y[-1]/10)
            state.append(self.evasion[j].theta[-1])
        
        return np.array(state)

    def plot(self, date_type, save):
        """ date_type = 'map', 'reward','win rate','dis' """
        def _moving_average(interval, windowsize):
            'Output smoothed polyline'
            window = np.ones(int(windowsize)) / float(windowsize)
            re = np.convolve(interval, window, 'valid')
            return re
        
        if date_type == 'map':
            plt.figure('map')
            self.map.plot_map()
            circle_theta = np.linspace(0, 2 * np.pi, 200)
            color = ['b','g','r','y']
            for i in range(NUM_PUR):
                plt.plot(self.pursuit[i].x, self.pursuit[i].y)
                for j in range(len(self.pursuit[i].x)):
                    if j % 10 == 0:
                        plt.scatter(self.pursuit[i].x[j], self.pursuit[i].y[j],c = color[i], marker='x')
                circle_x = CAPTURE_DISTANCE*np.cos(circle_theta) + self.pursuit[i].x[-1]
                circle_y = CAPTURE_DISTANCE*np.sin(circle_theta) + self.pursuit[i].y[-1]
                plt.plot(circle_x,circle_y,color="darkred", linewidth=2)
                plt.text(self.pursuit[i].x[0], self.pursuit[i].y[0], 'pur%s'%i)
            plt.plot(self.evasion[0].x, self.evasion[0].y)
            plt.scatter(self.evasion[0].x[-1], self.evasion[0].y[-1], marker='o')
            plt.text(self.evasion[0].x[0], self.evasion[0].y[0], 'eva')
            plt.xlim(-10, 170)
            plt.ylim(-10, 110)
            if save == True:
                plt.savefig('./results/map_time_%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
                plt.close('map')

        if date_type == 'reward':
            plt.figure('purs reward')
            for i in range(NUM_PUR):
                plt.subplot(2,2,i+1)
                plt.plot(range(len(self.reward_record_all[i])),self.reward_record_all[i])
                date = _moving_average(self.reward_record_all[i],50)
                plt.plot(range(len(date)),date)
            if save == True:
                plt.savefig('./results/reward_%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
                plt.close('purs reward')

        if date_type == 'time':
            plt.figure('game time')
            plt.plot(range(len(self.game_time)), self.game_time)
            date = _moving_average(self.game_time,50)
            plt.plot(range(len(date)),date)
            if save == True:
                plt.savefig('./results/game_time_%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
                plt.close('game time')

            plt.figure('catch time')
            plt.plot(range(len(self.pur_success_time)), self.pur_success_time)
            date = _moving_average(self.pur_success_time,50)
            plt.plot(range(len(date)),date)
            if save == True:
                plt.savefig('./results/catch_time_%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
                plt.close('catch time')



