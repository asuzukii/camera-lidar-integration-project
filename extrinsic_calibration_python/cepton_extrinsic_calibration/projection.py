###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
import open3d as o3d

from .linear_regression_L1 import L1_regression
from .data_io import *

def plane_detection_RANSAC(point_cloud):
  """
  Plane detection with RANSAC.
  Input:
    point_cloud [ndarray]: input point cloud of a single target/cluster. Array 
      of size (n, 4). The last column, intensity, won't be useful in this 
      function. 
  Return:
    a, b, c, d: the parameter for a 3D plane. ax + by + cz + d = 0.
  """

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

  plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                        ransac_n=10,
                                        num_iterations=1000)
  [a, b, c, d] = plane_model

  return a, b, c, d

def plane_detection_Huber(point_cloud):
  """
  Huber regressor
  """
  reg = linear_model.HuberRegressor().fit(point_cloud[:, 1:3], point_cloud[:, 0])

  b, c = reg.coef_[0], reg.coef_[1]
  d = reg.intercept_
  a = -1

  return a, b, c, d

def plane_detection_L1(point_cloud, outer_scale):
  """
  Plane detection with linear regression (L1 loss)
  """
  param = L1_regression(point_cloud[:, :3], np.zeros(len(point_cloud)), outer_scale)

  return param, 1

def points_to_image(data_2d, resolution=0.005):
  """
  Transform a set of 2D points to a 2D image. TODO: better interpolation method as mentioned by Mark
  Input:
    data_2d [ndarray]: Input data. Array of size (n, 3).
    resolution [float]: Quantization resolution .
  Return:
    2D image.
  """
  intensity = np.zeros([
                    int(np.max(data_2d, 0)[0] // resolution) + 1, 
                    int(np.max(data_2d, 0)[1] // resolution) + 1
              ])

  grid_dict = {}

  for point in data_2d:
    x = point[0]
    y = point[1]

    x_grid = int(x // resolution)
    y_grid = int(y // resolution)

    if not x_grid in grid_dict.keys():
        grid_dict[x_grid] = {}
    if not y_grid in grid_dict[x_grid].keys():
        grid_dict[x_grid][y_grid] = []

    grid_dict[x_grid][y_grid].append(point)

  for x_index in range(intensity.shape[0]):
    for y_index in range(intensity.shape[1]):
      if not x_index in grid_dict.keys():
          intensity[x_index, y_index] = 0
          continue

      if not y_index in grid_dict[x_index].keys():
          intensity[x_index, y_index] = 0
          continue

      points_in_grid = np.array(grid_dict[x_index][y_index])
      intensity[x_index, y_index] = np.mean(points_in_grid, 0)[2]

  return intensity
