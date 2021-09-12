import math
from math import sin,cos,tan 
g = 9.8
L = 0.5
MACH = 343
TIME = 0.1

class sixDOF(object):
    """
    Some Variable Definitions:
    n_x: Throttle acceleration directed out the nose of the aircraft in g's
    vel: Airspeed in meters/second
    gamma: Flight path angle in radians
    x,y,z: position in NED coordinates in meters where altitude h = -z
    phi: Roll angle in radians
    psi: Horizontal azimuth angle in radians
    alpha: Angle of attack in radians with respect to wind reference frame

    Our control input:
    thrust n_x , 0-7.5
    rate of change of angle of attack d_alpha -0.5-0.5
    the rate of change of the roll angle d_phi -1-1

    Above model is from the paper:
    An Efficient Algorithm for Multiple-Pursuer-Multiple-Evader Pursuit/Evasion Game
    """

    def __init__(self,initial_x,initial_y,initial_z):
        self.x = [initial_x]
        self.y = [initial_y]
        self.z = [initial_z]
        self.vel = [280]
        self.alpha = [0]
        self.phi = [0]
        self.gamma = [0]
        self.psi = [0]
        
        self.VEL_MIN = 0.1* MACH
        self.VEL_MAX = 0.35* MACH
        self.PSI_D1_MIN = -1.5
        self.PSI_D1_MAX = 1.5
        self.ALPHA_MIN = -0.009
        self.ALPHA_MAX = 0.69

    def move(self,alpha_d1,phi_d1,n_x):
        self.phi.append(self.phi[-1] + phi_d1*TIME)
        self.alpha.append(self._alpha_limit(self.alpha[-1] + alpha_d1*TIME))
        n_f = n_x * sin(self.alpha[-1]) + L

        vel_d1 = g* (n_x* cos(self.alpha[-1]) - sin(self.gamma[-1]))
        gamma_d1 = g/self.vel[-1]* (n_f * cos(self.phi[-1]) - cos(self.gamma[-1]))
        psi_d1 = g* (n_f * sin(self.phi[-1])/ (self.vel[-1]* cos(self.gamma[-1])))

        self.gamma.append(self.gamma[-1] + gamma_d1*TIME)
        self.psi.append(self.psi[-1] + self._psi_d1_limit(psi_d1*TIME))
        self.vel.append(self._vel_limit(self.vel[-1] + vel_d1*TIME))

        x_d1 = self.vel[-1] * cos(self.gamma[-1]) * cos(self.psi[-1])
        y_d1 = self.vel[-1] * cos(self.gamma[-1]) * sin(self.psi[-1])
        z_d1 = self.vel[-1] * sin(self.gamma[-1])

        self.x.append(self.x[-1] + x_d1*TIME)
        self.y.append(self.y[-1] + y_d1*TIME)
        self.z.append(self.z[-1] + z_d1*TIME)


    def _vel_limit(self,vel):
        """
        limit the velocity of aircraft
        """
        if vel >= self.VEL_MAX:
            return self.VEL_MAX
        elif vel <= self.VEL_MIN:
            return self.VEL_MIN
        else:
            return vel
    
    def _psi_d1_limit(self,psi_d1):
        """
        limit the psi_d1
        """
        if psi_d1 >= self.PSI_D1_MAX:
            return self.PSI_D1_MAX
        elif psi_d1 <= self.PSI_D1_MIN:
            return self.PSI_D1_MIN
        else:
            return psi_d1
        
    def _alpha_limit(self,alpha):
        """
        limit the alpha of aircraft
        """
        if alpha >= self.ALPHA_MAX:
            return self.ALPHA_MAX
        elif alpha <= self.ALPHA_MIN:
            return self.ALPHA_MIN
        else:
            return alpha
    
    def get_position(self):
        pass

    
if __name__ == "__main__":
    pass

