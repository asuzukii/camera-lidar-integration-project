###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class BezierModel:

  def __init__(self):
    self.source = None
    self.target = None
    self.NUM_DISTORTION_PARAM = 10
    self.parameters = None

  def load_input(self, source, target):
    self.source = source
    self.target = target

  def load_parameter(self, parameters:list):
    if len(parameters) > 16:
      self.parameters = parameters[2:10] + parameters[12:20]
    else:
      self.parameters = parameters

  def beinstein3rd(self, t, p1, p2, p3, p4):
    x = (1 - t) / 2
    y = (1 + t) / 2
    f1 = x**3
    f2 = 3 * x**2 * y
    f3 = 3 * x * y**2
    f4 = y**3
    return p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4

  def layer(self, pars, x, z):
    # Requirement on the firmware/hardware check with them
    x1 = x / 2
    z1 = z / 2
    
    x2 = self.beinstein3rd(x1, pars[0], pars[1], pars[2], pars[3]) * self.beinstein3rd(z1, pars[4], pars[5], pars[6], pars[7])
    z2 = self.beinstein3rd(x1, pars[8], pars[9], pars[10], pars[11]) * self.beinstein3rd(z1, pars[12], pars[13], pars[14], pars[15])
    return x2, z2

  def forward(self, pars, x_image, z_image):
    delta_x, delta_z = self.layer(pars, x_image, z_image)
    undistort_x = x_image + delta_x
    undistort_z = z_image + delta_z
    return undistort_x, undistort_z

  def cost_function_raw(self, pars, source, target):
    source_x = source[:, 0]
    source_z = source[:, 1]

    target_x = target[:, 0]
    target_z = target[:, 1]

    undistort_x, undistort_z = self.forward(pars, source_x, source_z)
    error = np.mean(np.abs(target_x - undistort_x) + np.abs(target_z - undistort_z))
    return error

  def optimize_bazier(self):
    if self.source is None or self.target is None:
      raise ValueError("No data input")
    #initial_guess = (0, 0, -2, -0.66667, 0.66667, 2, -2, -0.66667, 0.66667, 2)
    initial_guess = np.ones(16) * 0.1#(0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    bounds = ((-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), 
              (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), 
              (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3))
    result = optimize.minimize(self.cost_function_raw, initial_guess, 
        bounds=bounds, args=(self.source, self.target), 
        method='L-BFGS-B', options={"disp": False})
    self.parameters = result.x

    undistort_x, undistort_z = self.predict(self.source)
    #x_angular_undistorted, z_angular_undistorted = np.arctan(undistort_x), np.arctan(undistort_z)

    target_x, target_z = self.target[:, 0], self.target[:, 1]

    error_x = np.arctan(undistort_x) - np.arctan(target_x)
    error_z = np.arctan(undistort_z) - np.arctan(target_z)

    std_x = np.std(error_x)
    std_z = np.std(error_z)
    
    x_mask = np.abs(error_x) < 3 * std_x
    z_mask = np.abs(error_z) < 3 * std_z

    print("x: {}/{}".format(np.sum(x_mask), len(x_mask)))
    print("z: {}/{}".format(np.sum(z_mask), len(z_mask)))
    print("overall: {}/{}".format(np.sum(z_mask & x_mask), len(z_mask)))

    inlier_mask = x_mask & z_mask

    self.source = self.source[inlier_mask, :]
    self.target = self.target[inlier_mask, :]

    initial_guess = np.ones(16) * 0.1#(0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    bounds = ((-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), 
              (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), 
              (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3))
    result = optimize.minimize(self.cost_function_raw, initial_guess, 
        bounds=bounds, args=(self.source, self.target), 
        method='L-BFGS-B', options={"disp": False})
    self.parameters = result.x


  def predict(self, data):
    source_x = data[:, 0]
    source_z = data[:, 1]
    undistort_x, undistort_z = self.forward(self.parameters, source_x, source_z)
    return undistort_x, undistort_z

  def draw_distortion_map(self):
    """
    Plot the distortion map
    """

    xx, zz = np.meshgrid(np.arange(-45, 45+1e-5, 0.1), np.arange(-15, 10+1e-5, 0.1))

    x_grid_angular = xx.flatten() * np.pi / 180
    z_grid_angular = zz.flatten() * np.pi / 180

    x_grid_tan = np.tan(x_grid_angular)
    z_grid_tan = np.tan(z_grid_angular)


    x_tan_undistorted, z_tan_undistorted = self.predict(np.stack([x_grid_tan, z_grid_tan], 1))

    x_angular_undistorted, z_angular_undistorted = np.arctan(x_tan_undistorted), np.arctan(z_tan_undistorted)

    delta_x_degree = (x_angular_undistorted - x_grid_angular)*180/np.pi

    delta_x_map = np.reshape(delta_x_degree, [251, 901])
    delta_x_map = (delta_x_map - delta_x_map[:, ::-1]) / 2

    delta_z_degree = (z_angular_undistorted - z_grid_angular)*180/np.pi

    delta_z_map = np.reshape(delta_z_degree, [251, 901])
    delta_z_map = (delta_z_map + delta_z_map[:, ::-1]) / 2

    fig, ax = plt.subplots(figsize=(15,12))
    plt.subplot(211)
    plt.imshow(delta_x_map, cmap='seismic', origin='lower')

    plt.xticks(np.arange(0, 900+1, 50), np.arange(-45, 45+1, 5))
    plt.yticks(np.arange(0, 250+1, 50), np.arange(-15, 10+1, 5))
    plt.title("distortion x angle [deg]")
    plt.xlabel("x angle [deg]")
    plt.ylabel("z angle [deg]")
    plt.colorbar()

    plt.subplot(212)
    plt.imshow(delta_z_map, cmap='seismic', origin='lower')
    plt.xticks(np.arange(0, 900+1, 50), np.arange(-45, 45+1, 5)) 
    plt.yticks(np.arange(0, 250+1, 50), np.arange(-15, 10+1, 5))
    plt.title("distortion z angle [deg]")
    plt.xlabel("x angle [deg]")
    plt.ylabel("z angle [deg]")
    plt.colorbar()

    return delta_x_map, delta_z_map