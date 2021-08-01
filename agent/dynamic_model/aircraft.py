import math
from math import sin,cos,tan 
g = 10
VELOCITY_MAX = 400
VELOCITY_MIN = 40
class AIRCRAFT(object):
    """
    The model of aircraft motion model with threedegree-of-freedom particle model
    The angle of attack and the side slip angle are ignored, 
    assuming that the velocity direction coincides with the body axis.

    In the ground coordinate system, the ox axis takes the east,    
    the oy axis takes the north, and the oz axis takes the vertical direction.
    """
    def __init__(self,initial_x,initial_y,initial_z):
        self.x = [initial_x]
        self.y = [initial_y]
        self.z = [initial_z]
        self.vel = [280]
        self.gamma = [0]
        self.psi = [0]

    def move(self,thrust_force,pitch_force,roll_angle):

        vel_vary = g* (thrust_force - sin(self.gamma[-1]))
        gamma_vary = g/self.vel[-1]* (pitch_force* cos(roll_angle))
        psi_vary = g*pitch_force*sin(roll_angle)/ (self.vel[-1]*cos(self.gamma[-1]))

        self.vel.append(self._speed_limit(self.vel[-1] + vel_vary))
        self.gamma.append(self.gamma[-1] + gamma_vary)
        self.psi.append(self.psi[-1] + psi_vary)

        x_vary  = self.vel[-1] * cos(self.gamma[-1]) * sin(self.psi[-1])
        y_vary  = self.vel[-1] * cos(self.gamma[-1]) * cos(self.psi[-1])
        z_vary  = self.vel[-1] * sin(self.gamma[-1])

        self.x.append(self.x[-1] + x_vary)
        self.y.append(self.y[-1] + y_vary)
        self.z.append(self.z[-1] + z_vary)

    def _speed_limit(self,vel_test):
        """
        limit the velocity of aircraft
        """
        if vel_test >= VELOCITY_MAX:
            return VELOCITY_MAX
        elif vel_test <= VELOCITY_MIN:
            return VELOCITY_MIN
        else:
            return vel_test

if __name__ == "__main__":
    pass

