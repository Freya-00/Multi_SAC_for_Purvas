import aircraft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random
import math
import sixDOF
import uavplane



if __name__ == "__main__":
    a = uavplane.UAVPLANE(0,0)
    b = uavplane.UAVPLANE(0,0)
    for i in range(100):
        theta_d = (random()-0.5)*math.pi/12
        a.move(theta_d)
        theta_5 = (random()-0.5)*math.pi/12
        b.move(theta_5)
    
    
    fig = plt.figure()
    plt.plot(a.x,a.y)
    plt.plot(b.x,b.y)
    plt.show()  