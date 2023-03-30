"""
===============================================================================
Parameters to create the Black Hole image
===============================================================================
@author: Eduard Larrañga - 2023
===============================================================================
"""

from numpy import pi
from schwarzschild import *
from thin_disk import *
from image_plane import image_plane

###############################################################################
############################## METRIC DEFINITION ##############################
###############################################################################
metric = g
geodesics = geodesics


###############################################################################
############################ BLACK HOLE PARAMETERS ############################
###############################################################################
M = 1 # Mass


###############################################################################
############################# DETECTOR PARAMETERS #############################
###############################################################################
D = 100*M
iota = pi/2.2
screen_side = 20*M
n_pixels = 300
detector = image_plane(D=D, iota = iota, s_side = screen_side, n_pixels = n_pixels)


###############################################################################
############################# ACCRETION STRUCTURE #############################
###############################################################################
R_min = 6*M
R_max = 20*M
acc_structure = thin_disk(R_min, R_max)


###############################################################################
############################### IMAGE FILENAME ################################
###############################################################################
filename = 'BlackHole.jpeg'



###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')
