
"""
Create the map for the pursuit evasion game
author: Jiebang
"""

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

    def collection_detection(self,x,y):
        if x < 0 or x > self.length:
            return True
        elif y < 0 or y > self.width:
            return True
        else:
            return False
    
    def _obstacle(self):
        """
        bulid obstacle of map
        """
        pass

    def draw_map(self):
        pass

if __name__ == "__main__":
    pass
