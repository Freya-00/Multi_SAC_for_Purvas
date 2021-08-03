#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: pureva2D.py
# Creat Time: 2021-06-01 20:19:30 星期二
# Version: 1.0

# Description: # The env for pur-eva game in 2D plane
# Last Modified time: 2021-06-07 10:39:02 星期一
# Last Modified time: 2021-06-14 21:41:39 星期一
import sys

sys.path.append("../code")

import math

import matplotlib.pyplot as plt
import numpy as np
from random import random
from agent.sac_2d_agent_alpha import PurEva_2D_Agent
from environment.map import PurEvaMap
from environment.reward import PurEva_2D_Reward

"""
Definition of hyperparameters
"""

CAPTURE_DISTANCE = 3 # Captured distance
MAX_STEP = 150

"""
Definition of Game Class
"""
class PurEva_2D_Game(object):
    """
    The game env class, pur and eva play in 2d plane.
    
    Parameyters:
    num_pur: the num of pursuits
    num_eva: the num of evasions
    map_length: the length of map
    map_width: the width of map
    MAX_STEP: the max step of one eposide

    RL parameters:
    state: x,y,v of all agents and dim is num *3
    action: the difference of v and dim is 1
    """
    def __init__(self,
                num_pur = 3,
                num_eva = 1,
                map_length = 100,
                map_width = 100,
                MAX_STEP = MAX_STEP):

        self.num_pur = num_pur
        self.num_eva = num_eva
        self.state_dim_purs = num_pur*3 + num_eva*3 + 3
        self.state_dim_evas = num_pur*3 + num_eva*3
        self.action_dim = 1
        self.vel_pur = 1.5
        self.vel_eva = 1.5
        self.max_step = MAX_STEP
        self.map = PurEvaMap(map_length,map_width)
        self.module_reward = PurEva_2D_Reward(self.vel_pur, self.vel_eva,
                                                CAPTURE_DISTANCE, 0.5,
                                                map_length, map_width)
        self.reward_record_purs = []
        self.reward__record_evas = []
        self.reward_one_eposide_purs = []
        self.reward_one_eposide_eva = []

        initial_pos_purs = [[20.0, 20.0], [50.0, 80.0], [80.0, 30.0]]
        initial_pos_eva = [50.0, 50.0]
        self.pursuit = [] # agent of pursuit gamers
        for i in range(num_pur):
            self.pursuit.append(
                PurEva_2D_Agent('pur%d'%i, self.state_dim_purs, self.action_dim,
                                initial_pos_purs[i][0], initial_pos_purs[i][1],
                                self.vel_pur, share_action=True))
            self.pursuit[i].dynamic_model.reset_theta(initial_pos_eva)
            self.reward_record_purs.append([])
            self.reward_one_eposide_purs.append(0)
        
        self.evasion = [] # agent of evasion gamers
        for j in range(num_eva):
            self.evasion.append(
                PurEva_2D_Agent('eva%d'%j, self.state_dim_evas, self.action_dim,
                                initial_pos_eva[0], initial_pos_eva[1],
                                self.vel_eva, policy_deterministic= True,
                                creat_network=True, ))
            self.reward__record_evas.append([])
            self.reward_one_eposide_eva.append(0)

    def initial_env(self):
        '''initial the env before gameing'''
        for pur in self.pursuit:
            pur.dynamic_model.reset_dynamic_model()
        for eva in self.evasion:
            eva.dynamic_model.reset_dynamic_model()
        
        self.reward_one_eposide_purs = [0,0,0]
        self.reward_one_eposide_eva = [0]

    def act_and_learn(self, t, eval_f = False):
        'core function for act with env and learn'
        
        'get action and state'
        state_for_policy = self._get_state_policy()
        action_pur, action_eva = self._get_actions(state_for_policy, eval = eval_f)
        state_share_action = self._get_state_purs(state_for_policy, action_pur, action_eva)

        'move'
        for i in range(self.num_pur):
            self.pursuit[i].dynamic_model.move(action_pur[i])
            self._border_limit(self.pursuit[i])

        for j in range(self.num_eva):
            self.evasion[j].dynamic_model.move(action_eva[j])
            self._border_limit(self.evasion[j])

        'get reward and next_state'
        state_next_for_policy = self._get_state_policy()
        action_next_pur, action_next_eva = self._get_actions(state_next_for_policy)
        state_next_share_action = self._get_state_purs(state_next_for_policy, action_next_pur, action_next_eva)
        done, reward_pur, reward_eva, results = self._get_reward_and_done(t)
        
        'record'
        for i in range(self.num_pur):
            self.reward_one_eposide_purs[i] += reward_pur[i]
        for j in range(self.num_eva):
            self.reward_one_eposide_eva[j] += reward_eva[j]
        
        'learn'
        self._update_policy_purs(state_share_action, action_pur, reward_pur, state_next_share_action, done, t)
        self._update_policy_evas(state_for_policy, action_eva, reward_eva, state_next_for_policy, done, t)
        
        if done == True:
            for i in range(self.num_pur):
                self.reward_record_purs[i].append(self.reward_one_eposide_purs[i])
            for j in range(self.num_eva):
                self.reward__record_evas[j].append(self.reward_one_eposide_eva[j])
        
        return reward_pur, done, results

    def set_random_position(self):
        for j in range(self.num_eva):
            pos_x = random() * 100
            self.evasion[j].dynamic_model.initial_x = pos_x
            pos_y = random() * 100
            self.evasion[j].dynamic_model.initial_y = pos_y
            initial_pos_eva = [pos_x,pos_y]
        
        for i in range(self.num_pur):
            pos_x = random() * 100
            self.pursuit[i].dynamic_model.initial_x = pos_x
            pos_y = random() * 100
            self.pursuit[i].dynamic_model.initial_y = pos_y
            self.pursuit[i].dynamic_model.reset_theta(initial_pos_eva)
        
    def _get_actions(self, state, eval = False):
        action_pur = []
        action_eva = []

        'get action'
        for i in range(self.num_pur):
            action_pur.append(self.pursuit[i].get_action(state, eval))
        
        for j in range(self.num_eva):
            action_eva.append(self.evasion[j].get_action(state))

        return action_pur, action_eva

    def _get_state_purs(self, state_policy, action_purs, action_eva):
        'both agents have same state space'
        state_purs = []
        for i in range(self.num_pur):
            state_purs.append(state_policy)
            for j in range(self.num_pur):
                if j != i:
                    state_purs[i] = np.append(state_purs[i], action_purs[j])
            state_purs[i] = np.append(state_purs[i], action_eva)
        return state_purs
    
    def _get_state_policy(self):
        'both agents have same state space'
        state_pur = []
        for i in range(self.num_pur):
            state_pur.append(self.pursuit[i].dynamic_model.x[-1]/10)
            state_pur.append(self.pursuit[i].dynamic_model.y[-1]/10)
            state_pur.append(self.pursuit[i].dynamic_model.theta[-1])
        sp = np.array(state_pur)

        state_eva = []
        for j in range(self.num_eva):
            state_eva.append(self.evasion[j].dynamic_model.x[-1]/10)
            state_eva.append(self.evasion[j].dynamic_model.y[-1]/10)
            state_eva.append(self.evasion[j].dynamic_model.theta[-1])
        se = np.array(state_eva)

        state = np.append(sp, se)
        return state
    
    def _get_reward_and_done(self,t):
        position_all = []
        for pur in self.pursuit:
            position_all.append(pur.dynamic_model.get_pos())
        position_all.append(self.evasion[-1].dynamic_model.get_pos())
        done, reward_pur, reward_eva, results = self.module_reward.return_reward(position_all,t)
        return done, reward_pur, reward_eva, results

    def _update_policy_purs(self, state, ac, rp, next_state, done, t):
        for i in range(self.num_pur):
            self.pursuit[i].memory.push(state[i], ac[i], rp[i], next_state[i], done) # Append transition to memory
            # if t %2 == 0:
            self.pursuit[i].update_policy()
    
    def _update_policy_evas(self, state, ac, re, next_state, done, t):
        for j in range(self.num_eva):
            self.evasion[j].memory.push(state, ac[j], re[j], next_state, done)
            if t %2 == 0:
                self.evasion[j].update_policy()
                # print('eva learned')

    def _border_limit(self,agent):
        tem_x = agent.dynamic_model.x[-1]
        tem_y = agent.dynamic_model.y[-1]
        if tem_x < 0:
            agent.dynamic_model.x[-1] = 0
        if tem_x > self.map.length:
            agent.dynamic_model.x[-1] = self.map.length
        if tem_y < 0:
            agent.dynamic_model.y[-1] = 0
        if tem_y > self.map.width:
            agent.dynamic_model.y[-1] = self.map.width

    def _cal_dis(self,a_pos,b_pos):
        'input [x,y]'
        return math.sqrt((a_pos[0]-b_pos[0])**2 +
                        (a_pos[1]-b_pos[1])**2)

    def _cal_andis(self,a_vel,b_vel):
        'input: angel of agent'
        andis = abs(a_vel - b_vel)
        andis = andis % (math.pi/2)
        return andis

    def plot(self, show_map = True, show_dis = False, show_reward = True, show_win_rate = True, save_fig = False):
        '''plot game'''
        
        def _moving_average(interval, windowsize):
            'Output smoothed polyline'
            window = np.ones(int(windowsize)) / float(windowsize)
            re = np.convolve(interval, window, 'valid')
            return re
        
        if show_map:
            plt.figure('map')
            circle_theta = np.linspace(0, 2 * np.pi, 200)
            color = ['b','g','r']
            for i in range(self.num_pur):
                plt.plot(self.pursuit[i].dynamic_model.x, self.pursuit[i].dynamic_model.y)
                for j in range(len(self.pursuit[i].dynamic_model.x)):
                    if j % 10 == 0:
                        plt.scatter(self.pursuit[i].dynamic_model.x[j], self.pursuit[i].dynamic_model.y[j],c = color[i], marker='x')
                # plt.scatter(self.pursuit[i].dynamic_model.x[0], self.pursuit[i].dynamic_model.y[0],marker='o')
                circle_x = CAPTURE_DISTANCE*np.cos(circle_theta) + self.pursuit[i].dynamic_model.x[-1]
                circle_y = CAPTURE_DISTANCE*np.sin(circle_theta) + self.pursuit[i].dynamic_model.y[-1]
                plt.plot(circle_x,circle_y,color="darkred", linewidth=2)
                plt.text(self.pursuit[i].dynamic_model.x[0], self.pursuit[i].dynamic_model.y[0], 'pur%s'%i)
            plt.plot(self.evasion[0].dynamic_model.x, self.evasion[0].dynamic_model.y)
            plt.scatter(self.evasion[0].dynamic_model.x[-1], self.evasion[0].dynamic_model.y[-1], marker='o')
            plt.text(self.evasion[0].dynamic_model.x[0], self.evasion[0].dynamic_model.y[0], 'eva')
            plt.xlim(-10, 110)
            plt.ylim(-10, 110)

        if show_dis:
            # plt.figure('distance')
            # plt.subplot(2,2,1)
            # plt.plot(range(len(self.dis_bw_pe[0])),self.dis_bw_pe[0])
            # plt.subplot(2,2,2)
            # plt.plot(range(len(self.dis_bw_pe[1])),self.dis_bw_pe[1])
            # plt.subplot(2,2,3)
            # plt.plot(range(len(self.dis_bw_pe[2])),self.dis_bw_pe[2])
            pass

        if show_reward:
            plt.figure('purs reward')
            plt.subplot(2,2,1)
            plt.plot(range(len(self.reward_record_purs[0])),self.reward_record_purs[0])
            date = _moving_average(self.reward_record_purs[0],50)
            plt.plot(range(len(date)),date)

            plt.subplot(2,2,2)
            plt.plot(range(len(self.reward_record_purs[1])),self.reward_record_purs[1])
            date = _moving_average(self.reward_record_purs[1],50)
            plt.plot(range(len(date)),date)
            
            plt.subplot(2,2,3)
            plt.plot(range(len(self.reward_record_purs[2])),self.reward_record_purs[2])
            date = _moving_average(self.reward_record_purs[2],50)
            plt.plot(range(len(date)),date)

            plt.figure('evas reward')
            plt.plot(range(len(self.reward__record_evas[0])),self.reward__record_evas[0])
            date = _moving_average(self.reward__record_evas[0],50)
            plt.plot(range(len(date)),date)
        # plt.savefig('./%s_%s_%d.jpg'%(data_name,man_position,time.time())) # save

        if show_win_rate:
            plt.figure('win rates')
            plt.plot(range(len(self.module_reward.win_rate)),self.module_reward.win_rate)
            plt.ylim(-0.1,1.1)



if __name__ == "__main__":
    EPOSIDES = 4000
    num_pur_win = 0
    num_eva_win = 0
    env = PurEva_2D_Game()
    for i in range(EPOSIDES):
        env.initial_env()
        reward = 0
        for j in range(env.max_step):
            re, done = env.act_and_learn(j)
            reward += re[-1]
            if done == True:
                break
            # env.plot(show_map = True, show_dis = False, show_reward = False, show_win_rate = False)
            # plt.show()
        print("epsidoe",i,"reward",reward/env.max_step)
        'plot'
        if i % 100  == 0 and i !=0:
            env.plot()
            plt.show()
        if i %10 == 0:
            env.initial_env()
            reward_test = 0
            for j in range(env.max_step):
                re, done = env.act_and_learn(j, eval_f = True)
                reward_test += re[-1]
                if done == True:
                    break
            print("epsidoe_test",i,"reward",reward/env.max_step)
    env.plot()
    plt.show()

