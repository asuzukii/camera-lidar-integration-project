###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import matplotlib.pyplot as plt
import numpy as np
import os
from .models.bezier import BezierModel


def plot_3D(points1, points2, title):
  """
  Make 3D plots
  Input:
    target: refers to ground-truth
    transformed_source: 
    title
  """

  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='b')
  ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='r')
  ax.set_title(title)
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

def parameter_distortion_map(distortion_param, save_dir = "./output/"):
  """
  Plot the distortion map
  """

  xx, zz = np.meshgrid(np.arange(-45, 45+1e-5, 0.5), np.arange(-15, 10+1e-5, 0.5))

  x_grid_angular = xx.flatten() * np.pi / 180
  z_grid_angular = zz.flatten() * np.pi / 180

  x_grid_tan = np.tan(x_grid_angular)
  z_grid_tan = np.tan(z_grid_angular)

  model = BezierModel()
  model.load_parameter(distortion_param)

  x_tan_undistorted, z_tan_undistorted = model.predict(np.stack([x_grid_tan, z_grid_tan], 1))

  x_angular_undistorted, z_angular_undistorted = np.arctan(x_tan_undistorted), np.arctan(z_tan_undistorted)

  delta_x_degree = (x_angular_undistorted - x_grid_angular)*180/np.pi

  delta_x_map = np.reshape(delta_x_degree, [51, 181])


  delta_z_degree = (z_angular_undistorted - z_grid_angular)*180/np.pi

  delta_z_map = np.reshape(delta_z_degree, [51, 181])

  fig, ax = plt.subplots(figsize=(15,12))
  plt.subplot(211)
  plt.imshow(delta_x_map, cmap='seismic', origin='lower')

  plt.xticks(np.arange(0, 180+1, 10), np.arange(-45, 45+1, 5)) 
  plt.yticks(np.arange(0, 50+1, 10), np.arange(-15, 10+1, 5))
  plt.title("distortion x angle [deg]")
  plt.xlabel("x angle [deg]")
  plt.ylabel("z angle [deg]")
  plt.colorbar()

  plt.subplot(212)
  plt.imshow(delta_z_map, cmap='seismic', origin='lower')

  plt.xticks(np.arange(0, 180+1, 10), np.arange(-45, 45+1, 5)) 
  plt.yticks(np.arange(0, 50+1, 10), np.arange(-15, 10+1, 5))
  plt.title("distortion z angle [deg]")
  plt.xlabel("x angle [deg]")
  plt.ylabel("z angle [deg]")
  plt.colorbar()

  if not save_dir is None:
    plt.savefig(os.path.join(save_dir, "distortion_map_xz.png"))
