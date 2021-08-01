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
import numpy as np
from environment.map import PurEvaMap
import matplotlib.pyplot as plt
from agent.sac_2d_agent_central import PurEva_2D_Agent
from environment.reward import PurEva_2D_Reward
"""
Definition of hyperparameters
"""

CAPTURE_DISTANCE = 3 # Captured distance
MAX_END_DISTANCE = 1.5 # ending distance 


class PurEva_2D_Game(object):
    """
    The game env class, pur and eva play in 2d plane
    
    Parameyters:
    pur_num: the num of pursuits
    eva_num: the num of evasions
    map_length: the length of map
    map_width: the width of map
    MAX_STEP: the max step of one eposide

    RL parameters:
    state: x,y,v of all agents and dim is num *3
    action: the difference of v and dim is 1
    """
    def __init__(self,
                pur_num = 3,
                eva_num = 1,
                map_length = 100,
                map_width = 100,
                MAX_STEP = 150):

        self.pur_num = pur_num
        self.eva_num = eva_num
        self.state_dim = pur_num*3 + eva_num*3
        self.action_dim = 1
        self.vel_pur = 1.0
        self.vel_eva = 2.0
        self.max_step = MAX_STEP
        self.map = PurEvaMap(map_length,map_width)
        self.curriculum_reward = PurEva_2D_Reward(
                                                self.vel_pur,
                                                self.vel_eva,
                                                CAPTURE_DISTANCE,
                                                0.5,
                                                map_length,map_width)
        self.rw_record_purs = []
        self.rw_record_evas = []
        self.rw_one_eposide_purs = []
        self.rw_one_eposide_eva = []
        self.win_rate = []

        initial_pos = [[20,20], [50,80], [80,30]]
        self.pursuit = [] # agent of pursuit gamers
        for i in range(pur_num):
            self.pursuit.append(
                PurEva_2D_Agent('pur%d'%i,self.state_dim,self.action_dim,
                                initial_pos[i][0],initial_pos[i][1],
                                self.vel_pur))
            self.pursuit[i].dynamic_model.reset_theta([50.0,50.0])
            self.rw_record_purs.append([])
            self.rw_one_eposide_purs.append(0)
        # self.pursuit[1].dynamic_model.initial_theta = 1.57
        
        self.evasion = [] # agent of evasion gamers
        for i in range(eva_num):
            self.evasion.append(
                PurEva_2D_Agent('eva%d'%i,self.state_dim,self.action_dim,
                                50,50,
                                self.vel_eva,
                                creat_network=True))
            self.rw_record_evas.append([])
            self.rw_one_eposide_eva.append(0)
        self.evasion[0].dynamic_model.turning_angle = 3
        
        self.dis_bw_pe = [] # distance between purs and evas
        self.andis_bw_pe = [] # angle distance between purs and evas
        for i in range(pur_num):
            for j in range(eva_num):
                """
                one array correspond a pur and his eva
                [[p1e1] [p1e2] [p1e3]
                 [p2e1] [p2e2] [p2e3]
                 [p3e1] [p3e2] [p3e3]]
                """
                self.dis_bw_pe.append([self._cal_dis(self.pursuit[i].dynamic_model.get_pos(),
                                                        self.evasion[j].dynamic_model.get_pos())])
                # self.andis_bw_pe.append([self._cal_andis(self.pursuit[i].dynamic_model.theta[-1],
                #                                                 self.evasion[j].dynamic_model.theta[-1])])

    def initial_env(self):
        'initial the env before gameing'
        for pur in self.pursuit:
            pur.dynamic_model.reset_dynamic_model()
        for eva in self.evasion:
            eva.dynamic_model.reset_dynamic_model()
        
        self.rw_one_eposide_purs = [0,0,0]
        self.rw_one_eposide_eva = [0]
        # for i in range(self.pur_num):
        #     self.rw_record_purs[i].append(0)
        # for j in range(self.eva_num):
        #     self.rw_record_evas[j].append(0)
        
        self.dis_bw_pe = [] # distance between purs and evas 
        self.andis_bw_pe = [] # angle distance between purs and evas
        for i in range(self.pur_num):
            for j in range(self.eva_num):
                self.dis_bw_pe.append([self._cal_dis(self.pursuit[i].dynamic_model.get_pos(),
                                                        self.evasion[j].dynamic_model.get_pos())])
                # self.andis_bw_pe.append([self._cal_andis(self.pursuit[i].dynamic_model.theta[-1],
                #                                                 self.evasion[j].dynamic_model.theta[-1])])

    def act_and_learn(self,state,t):
        'core function for act with env and learn'
        action_pur = []
        for i in range(self.pur_num):
            _acp = self.pursuit[i].get_action(state)
            action_pur.append(_acp)
            self.pursuit[i].dynamic_model.move(_acp)
            self._border_limit(self.pursuit[i])

        action_eva = []
        for j in range(self.eva_num):
            _ace = self.evasion[j].get_action(state)
            action_eva.append(_ace)
            self.evasion[j].dynamic_model.move(_ace)
            self._border_limit(self.evasion[j])

        self._update_distance() # update dis and andis
        state_next = self.get_state()
        done, dflag = self._done_judge(t)
        reward_pur = self._reward_of_purs()
        reward_eva = self._reward_of_evas(dflag,t)
        
        for i in range(self.pur_num):
            self.rw_one_eposide_purs[i] += reward_pur[i]
        for j in range(self.eva_num):
            self.rw_one_eposide_eva[j] += reward_eva[j]
        
        self._update_policy_purs(state, action_pur, reward_pur, state_next, done,t)
        self._update_policy_evas(state, action_eva, reward_eva, state_next, done,t)
        
        if done == True:
            for i in range(self.pur_num):
                self.rw_record_purs[i].append(self.rw_one_eposide_purs[i])
            for j in range(self.eva_num):
                self.rw_record_evas[j].append(self.rw_one_eposide_eva[j])
        
        return reward_pur, state_next, done

    def get_state(self):
        'both agents have same state space'
        state_pur = []
        for i in range(self.pur_num):
            state_pur.append(self.pursuit[i].dynamic_model.x[-1]/10)
            state_pur.append(self.pursuit[i].dynamic_model.y[-1]/10)
            state_pur.append(self.pursuit[i].dynamic_model.theta[-1])
        sp = np.array(state_pur)

        state_eva = []
        for j in range(self.eva_num):
            state_eva.append(self.evasion[j].dynamic_model.x[-1]/10)
            state_eva.append(self.evasion[j].dynamic_model.y[-1]/10)
            state_eva.append(self.evasion[j].dynamic_model.theta[-1])
        se = np.array(state_eva)

        state = np.append(sp, se)
        return state

    def _reward_of_purs(self):
        position_all = []
        for pur in self.pursuit:
            position_all.append(pur.dynamic_model.get_pos())
        position_all.append(self.evasion[-1].dynamic_model.get_pos())
        # print('position_all = ',position_all)
        re_purs = self.curriculum_reward.return_reward(position_all)
        # print('re_purs = ',re_purs)
        return re_purs

    def _reward_of_evas(self,dflag,time):
        re_evas= []
        dis_recent = []
        for a in self.dis_bw_pe:
            dis_recent.append(a[-1])
        dis_recent = np.array(dis_recent)
        dis_min = dis_recent.min() # min distance
        for j in range(self.eva_num):
            r = math.log(dis_min/10)
            # r = time/30
            re_evas.append(r)
            # self.rw_record_evas[j][-1] += r
        return re_evas

    def _done_judge(self,t):
        global num_pur_win
        global num_eva_win
        dis_recent = []
        done = False
        d_flag = 3
        for a in self.dis_bw_pe:
            dis_recent.append(a[-1])
        dis_recent = np.array(dis_recent)
        dis_min = dis_recent.min()
        if dis_min <= CAPTURE_DISTANCE:
            num_pur_win += 1
            print('pur win',num_pur_win)
            done = True
            d_flag = 1
            self.win_rate.append(num_pur_win/(num_eva_win + num_pur_win))
        elif t >= self.max_step - 1:
            num_eva_win += 1
            print('eva win',num_eva_win)
            done = True
            d_flag = 2
            self.win_rate.append(num_pur_win/(num_eva_win + num_pur_win))
        return done, d_flag

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

    def _update_policy_purs(self,state, ac, rp, next_state, done, t):
        for i in range(self.pur_num):
            self.pursuit[i].net.push_data_to_buffer(state, ac[i], rp[i], next_state, done)
            if len(self.pursuit[i].net.replay_buffer) > self.pursuit[i].net.batch_size and t%3 ==0:
                self.pursuit[i].net.networks_update()
    
    def _update_policy_evas(self,state, ac, re, next_state, done, t):
        for j in range(self.eva_num):
            self.evasion[j].net.push_data_to_buffer(state, ac[j], re[j], next_state, done)
            if len(self.evasion[j].net.replay_buffer) > self.evasion[j].net.batch_size and t%3 ==0:
                self.evasion[j].net.networks_update()
    
    def _update_distance(self):
        pos_pur = []
        # ag_pur = []
        for i in range(self.pur_num):
            pos_pur.append(self.pursuit[i].dynamic_model.get_pos())
            # ag_pur.append(self.pursuit[i].dynamic_model.theta[-1])
        
        pos_eva = []
        # ag_eva = []
        for j in range(self.eva_num):
            pos_eva.append(self.evasion[j].dynamic_model.get_pos())
            # ag_eva.append(self.evasion[j].dynamic_model.theta[-1])
        
        for i in range(self.pur_num):
            for j in range(self.eva_num):
                self.dis_bw_pe[i+j].append(self._cal_dis(pos_pur[i],pos_eva[j]))
                # self.andis_bw_pe[i+j].append(self._cal_andis(ag_pur[i],ag_eva[j]))

    def _cal_dis(self,a_pos,b_pos):
        'input [x,y]'
        return math.sqrt((a_pos[0]-b_pos[0])**2 +
                        (a_pos[1]-b_pos[1])**2)

    def _cal_andis(self,a_vel,b_vel):
        'input: angel of agent'
        andis = abs(a_vel - b_vel)
        andis = andis % (math.pi/2)
        return andis

    def plot_map(self):
        plt.figure('map')
        circle_theta = np.linspace(0, 2 * np.pi, 200)
        color = ['b','g','r']
        for i in range(self.pur_num):
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
        plt.scatter(self.evasion[0].dynamic_model.x[-1], self.evasion[0].dynamic_model.x[-1], marker='o')
        plt.text(self.evasion[0].dynamic_model.x[0], self.evasion[0].dynamic_model.y[0], 'eva')
        plt.xlim(-10, 110)
        plt.ylim(-10, 110)

    def plot_dis(self):
        plt.figure('distance')
        plt.subplot(2,2,1)
        plt.plot(range(len(self.dis_bw_pe[0])),self.dis_bw_pe[0])
        plt.subplot(2,2,2)
        plt.plot(range(len(self.dis_bw_pe[1])),self.dis_bw_pe[1])
        plt.subplot(2,2,3)
        plt.plot(range(len(self.dis_bw_pe[2])),self.dis_bw_pe[2])

    def plot_andis(self):
        pass

    def plot_reward(self):
        plt.figure('purs reward')
        plt.subplot(2,2,1)
        # print(self.rw_record_purs)
        plt.plot(range(len(self.rw_record_purs[0])),self.rw_record_purs[0])
        date = self._moving_average(self.rw_record_purs[0],10)
        plt.plot(range(len(date)),date)

        plt.subplot(2,2,2)
        plt.plot(range(len(self.rw_record_purs[1])),self.rw_record_purs[1])
        date = self._moving_average(self.rw_record_purs[1],10)
        plt.plot(range(len(date)),date)
        
        plt.subplot(2,2,3)
        plt.plot(range(len(self.rw_record_purs[2])),self.rw_record_purs[2])
        date = self._moving_average(self.rw_record_purs[2],10)
        plt.plot(range(len(date)),date)

        plt.figure('evas reward')
        plt.plot(range(len(self.rw_record_evas[0])),self.rw_record_evas[0])
        date = self._moving_average(self.rw_record_evas[0],10)
        plt.plot(range(len(date)),date)
        # plt.savefig('./%s_%s_%d.jpg'%(data_name,man_position,time.time())) # save
    
    def plot_win_rate(self):
        plt.figure('win rates')
        plt.plot(range(len(self.win_rate)),self.win_rate)
        plt.ylim(-0.1,1.1)

    def _moving_average(self, interval, windowsize):
        'Output smoothed polyline'
        window = np.ones(int(windowsize)) / float(windowsize)
        re = np.convolve(interval, window, 'valid')
        return re


if __name__ == "__main__":
    EPOSIDES = 5000
    num_pur_win = 0
    num_eva_win = 0
    env = PurEva_2D_Game()
    # print(env.get_state())
    # print(type(env.get_state()))
    ep_reward = []
    for i in range(EPOSIDES):
        env.initial_env()
        reward = 0
        for j in range(env.max_step):
            state = env.get_state()
            re, state_next, done = env.act_and_learn(state,j)
            reward += re[-1]
            # env.plot_dis()
            # env.plot_map()
            # # env.plot_reward()
            # plt.show()
            if done == True:
                break
        ep_reward.append(reward/env.max_step)
        # print(env.dis_bw_pe[1,:][-1])
        # print(env.dis_bw_pe)
        print("epsidoe",i,"reward",reward/env.max_step)
        'plot'
        if i% 300 == 0 :
        # if i:
            env.plot_dis()
            env.plot_map()
            env.plot_reward()
            if i >10:
                env.plot_win_rate()
            plt.show()
    env.plot_dis()
    env.plot_map()
    env.plot_reward()
    env.plot_win_rate()
    plt.show()

