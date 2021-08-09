
"""
Create the map for the pursuit evasion game
author: Jiebang
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

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
    def __init__(self,length,width):
        self.length = length
        self.width = width
        self.pos_obstacle = [[length/4, width/4],
                            [3*length/4, width/4], 
                            [length/4, 3*width/4], 
                            [3*length/4, 3*width/4]]
        self.radius_obstacle = [15,15,15,15]

    def collection_detection(self,x,y):
        if x < 0 or x > self.length:
            return True
        elif y < 0 or y > self.width:
            return True
        else:
            return False
    
    def generate_obstacle(self):
        """
        bulid obstacle of map
        """
        def _cal_distance(a_pos,b_pos):
            return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
        
        center_a = [random.random()*40+30,random.random()*40+30]
        center_b = []
        for i in range(3):
            center_b_tem = [random.random()*80+10,random.random()*80+10]
            while _cal_distance(center_b_tem,center_a) <= 25:
                center_b_tem = [random.random()*80+10,random.random()*80+10]
            center_b.append(center_b_tem)
        
        return center_a, center_b

    def polt_obstacle(self):
        circle_theta = np.linspace(0, 2 * np.pi, 200)
        circle_x = self.radius_obstacle[0]*np.cos(circle_theta) + self.pos_obstacle[0][0]
        circle_y = self.radius_obstacle[0]*np.sin(circle_theta) + self.pos_obstacle[0][1]
        plt.plot(circle_x,circle_y,color="darkred", linewidth=2)
        for i in range(3):
            circle_x = self.radius_obstacle[i+1]*np.cos(circle_theta) + self.pos_obstacle[i+1][0]
            circle_y = self.radius_obstacle[i+1]*np.sin(circle_theta) + self.pos_obstacle[i+1][1]
            plt.plot(circle_x,circle_y,color="darkred", linewidth=2)
        # plt.show()
        # plt.xlim(0,100)
        # plt.ylim(0,100)
    
    def draw_map(self):
        pass

    def obstacle_collision_detection(self, pos):
        collision_flag = False
        def _cal_distance(a_pos,b_pos):
            return math.sqrt((a_pos[0]-b_pos[0])**2 + (a_pos[1]-b_pos[1])**2)
        for i in range(4):
            if _cal_distance(self.pos_obstacle[i], pos) <= self.radius_obstacle[i]:
                collision_flag = True
                break
        if pos[0] >= self.length or pos[0] <=0 or pos[1] <= 0 or pos[1] >= self.width:
            collision_flag = True
        return collision_flag

if __name__ == "__main__":
    map = PurEvaMap(100,100)
    map.polt_obstacle()
    plt.show()
