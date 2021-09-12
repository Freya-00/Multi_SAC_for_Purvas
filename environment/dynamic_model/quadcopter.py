"""
author: Jiebang
email: xingjiebang@gamil.com
A quadcopter model for reinforcement learning
"""
import math
from math import sin,cos,tan 
from random import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
g = 9.81

class QUADCOPTER(object):

    def __init__(self,initial_x,initial_y,initial_z):
        self.x = [initial_x]
        self.y = [initial_y]
        self.z = [initial_z]
        self.x_d1 = [0]
        self.y_d1 = [0]
        self.z_d1 = [0]
        self.phi = [0]
        self.theta = [0]
        self.psi = [0]
        self.phi_d1 = [0]
        self.theta_d1 = [0]
        self.psi_d1 = [0]
        self.kt = 0.000032 # thrust coefficient
        self.km = 0.00000075 # moment coefficient
        self.length = 0.25 # The moment arm of one arm length of the quadcopter
        self.mass = 1.4
        self.Ixx = 0.0075
        self.Iyy = 0.0075
        self.Izz = 0.0130
        self.VMAX_x = 14
        self.VMAX_y = 14
        self.VMAX_z = 5

    def move(self,omega1,omega2,omega3,omega4):
        U1 = self.kt*(omega1**2 + omega2**2 + omega3**3 + omega4**2)
        U2 = self.length* self.kt* (-omega1**2 + omega3**2)
        U3 = self.length* self.kt* (-omega2**2 + omega3**4)
        U4 = self.km* (omega1**2 - omega2**2 + omega3**3 - omega4**2)

        x_d2 = (sin(self.psi[-1])* sin(self.phi[-1])
                + cos(self.phi[-1])* sin(self.theta[-1])* cos(self.psi[-1])
                )* U1/self.mass
        y_d2 = (-cos(self.psi[-1]* sin(self.phi[-1]))
                + cos(self.phi[-1])* sin(self.theta[-1])* sin(self.psi[-1])
                )* U1/self.mass
        z_d2 = -g + (cos(self.phi[-1])* cos(self.theta[-1]))* U1/self.mass
        
        phi_d2 = (self.Iyy - self.Izz)/ self.Ixx * self.theta_d1[-1]* self.psi_d1[-1]\
                    + U2/ self.Ixx
        theta_d2 = (self.Izz - self.Ixx)/ self.Iyy * self.psi_d1[-1]* self.phi_d1[-1]\
                    + U3/ self.Iyy
        psi_d2 = (self.Ixx - self.Iyy)/ self.Izz * self.phi_d1[-1]* self.theta_d1[-1]\
                    + U4/ self.Izz
        psi_d2 = round(psi_d2,2)
        self.phi_d1.append(self.phi_d1[-1] + phi_d2)
        self.theta_d1.append(self.theta_d1[-1] + theta_d2)
        self.psi_d1.append(self.psi_d1[-1] + psi_d2)

        self.phi.append(self.phi[-1] + self.phi_d1[-1])
        self.theta.append(self.theta[-1] + self.theta_d1[-1])
        self.psi.append(self.psi[-1] + self.psi_d1[-1])

        self.x_d1.append(self._speed_limit(self.x_d1[-1] + x_d2,'x'))
        self.y_d1.append(self._speed_limit(self.y_d1[-1] + y_d2,'y'))
        self.z_d1.append(self._speed_limit(self.y_d1[-1] + y_d2,'z'))

        self.x.append(self.x[-1] + self.x_d1[-1])
        self.y.append(self.y[-1] + self.y_d1[-1])
        self.z.append(self.z[-1] + self.z_d1[-1])


    def _speed_limit(self,vel_test,axis):
        """
        limit the velocity of aircraft
        """
        if axis == 'x':
            VELOCITY_MAX = self.VMAX_x
        elif axis == 'y':
            VELOCITY_MAX = self.VMAX_y
        else:
            VELOCITY_MAX = self.VMAX_z

        if vel_test >= VELOCITY_MAX:
            return VELOCITY_MAX
        else:
            return vel_test

if __name__ == "__main__":
    a = AIRCRAFT(0,0,0)
    b = 10
    for i in range(50):
        # u1 = random()*b
        # u2 = random()*b
        # u3 = random()*b
        # u4 = random()*b
        u1 = 5.0
        u2 = 5.0
        u3 = 5.0
        u4 = 5.0
        a.move(u1,u2,u3,u4)
    # print("x= ",a.x)
    # print("y= ",a.y)
    # print("z= ",a.z)
    # print("phi= ",a.phi)
    # print("theta= ",a.theta)
    print("psi_d1 = ",a.psi_d1)
    print("psi = ",a.psi)
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(a.x,a.y,a.z,'gray')    #绘制空间曲线
    plt.show()

