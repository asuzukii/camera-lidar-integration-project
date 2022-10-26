###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np

# Coordinate change from VICON frame to sensor frame 
# Same as rotation about z clockwise
rot_VICON_to_sensor = np.array(
                      [
                        [0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1],
                      ]
                    )
# Coordinate change from sensor frame to VICON frame 
# Same as rotation about z anti-clockwise
rot_sensor_to_VICON = np.array(
                      [
                        [0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1],
                      ]
                    ) 

def coord_rotation_VICON_to_sensor(input):
  """
  Coordinate change from VICON to sensor
  input [ndarray] [3, n]
  """
  return (rot_VICON_to_sensor @ input.T).T

def coord_rotation_sensor_to_VICON(input):
  """
  Coordinate change from VICON to sensor
  input [ndarray] [3, n]
  """
  return (rot_sensor_to_VICON @ input.T).T