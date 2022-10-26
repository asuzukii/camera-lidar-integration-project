###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
from .distortion import to_image, apply_bezier_distortion
from .bezier import BezierModel
from scipy.optimize import minimize
import transforms3d

class JointModel:
  def __init__(self):
    self.source = None
    self.target = None
    self.source_img = None
    self.target_img = None
    self.parameters = None
    self.bezier = BezierModel()

  def optimize(self):
    self.parameters = self.optimize_RTD(self.source, self.target, param=self.parameters,  
                max_iter=500, display=True, alpha=1e-5)

  def load_parameters(self, R, t, distortion_param):
    euler = transforms3d.euler.mat2euler(R)
    euler = np.array(euler)

    self.parameters = np.concatenate([euler, t, distortion_param])

  def load_input(self, source, target):
    self.source = source
    self.target = target
    self.source_img = to_image(self.source)
    self.target_img = to_image(self.target)

  def get_parameters(self):
    rotation = self.parameters[:3]
    translation = self.parameters[3:6]
    distortion = self.parameters[6:]
    R = transforms3d.euler.euler2mat(*rotation)
    return R, translation, distortion

  def optimize_RTD(self, source, target, param, max_iter=500, tol=1e-5, alpha=1e-5, display=False):
    """
    Optimize Rotation, Translation and Distortion
    """
    n = len(source)

    parameters = param.copy()

    euler = parameters[:3]
    trans = parameters[3:6]

    bounds = np.tile([-np.inf, np.inf], (len(parameters), 1))
    bounds[:3, 0] = -np.pi
    bounds[:3, 1] = np.pi

    opt_res = minimize(
        self.RTD_loss_L1, parameters, 
        args=(source, target, alpha, trans, euler),
        options={"maxiter": max_iter, "gtol": tol, "disp":display},
        bounds=bounds)
    parameters = opt_res.x

    return parameters

  def RTD_loss_L1(self, w, source, target, alpha, trans, euler):
    """
    RTD means rotation, translation and distortion
    Compute the loss of current set of parameters
    """

    rotation = w[:3]
    translation = w[3:6]
    distortion = w[6:]
    R = transforms3d.euler.euler2mat(*rotation)

    self.bezier.parameters = distortion

    source_undistorted = apply_bezier_distortion(self.source, self.bezier)

    source_undistorted_transformed = (R@source_undistorted.T + translation[:, None]).T

    abs_loss = np.mean(np.linalg.norm(source_undistorted_transformed - self.target, 2, 1)) + \
                      10 * alpha * np.sum(np.abs(trans - translation)) + \
                      10 * alpha * np.sum(np.abs(euler - rotation))
    return abs_loss