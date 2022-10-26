###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import transforms3d

POLYNOMIAL_ORDER_MAX = 5
NUM_DISTORTION_PARAM_MAX = (POLYNOMIAL_ORDER_MAX+1)*(POLYNOMIAL_ORDER_MAX+2) // 2

POLYNOMIAL_ORDER = 4
NUM_DISTORTION_PARAM = (POLYNOMIAL_ORDER+1)*(POLYNOMIAL_ORDER+2) // 2

def nominal_parameters():

  return {"transformation": np.eye(4), "distortion": np.array([0,0,-2,-2/3,2/3,2,-2,-2/3,2/3,2])}

def inverse_parameters(parameters):
  """
  Get inverse parameters. Distortion and scale or offset will not be inversed.
  """
  inverse_parameters = np.zeros_like(parameters)
  translation, rotation, distortion, w_x, w_z, scale, offset = \
      parameter_unpack(parameters, False, False)
  
  transformation_matrix = np.eye(4)
  transformation_matrix[:3, :3] = transforms3d.euler.euler2mat(*rotation)
  transformation_matrix[:3, 3] = translation
  inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

  inverse_rotation = transforms3d.euler.mat2euler(inverse_transformation_matrix[:3, :3])
  inverse_translation = inverse_transformation_matrix[:3, 3]

  inverse_parameters[:3] = inverse_translation
  inverse_parameters[3:6] = inverse_rotation

  # Scale will be set to 1 by default
  inverse_parameters[-2] = 1.

  return inverse_parameters

def parameter_unpack(w, apply_scale, apply_offset):
  # Parameter unpack

  if len(w) == 2*NUM_DISTORTION_PARAM + 6 + 2:
    translation = w[:3]
    rotation = w[3:6]*np.pi
    distortion = w[6:6+2*NUM_DISTORTION_PARAM]
    w_x = distortion[:NUM_DISTORTION_PARAM].copy()
    w_x[1] += 1
    w_z = distortion[NUM_DISTORTION_PARAM:NUM_DISTORTION_PARAM*2].copy()
    w_z[2] += 1
    
    if apply_scale:
      scale = w[6+NUM_DISTORTION_PARAM*2]
    else:
      scale = 1.
    if apply_offset:
      offset = w[6+NUM_DISTORTION_PARAM*2+1]
    else:
      offset = 0
  else:
    raise ValueError("Unknown parameters")

  return translation, rotation, distortion, w_x, w_z, scale, offset

def apply_bezier_distortion(source, model):
  """
  Given a set of distortion parameters, apply it to the point cloud
  """

  # Do un-distortion in image coordinate
  source_image, _ = to_image(source)
  x_image_undistorted, z_image_undistorted = model.predict(source_image)

  x_image_undistorted = x_image_undistorted.astype(np.float32)
  z_image_undistorted = z_image_undistorted.astype(np.float32)

  # Transfrom from image coordinate to Cartesian coordinate
  r2 = (np.linalg.norm(source.astype(np.float32), 2, 1))**2
  x_undistorted = np.sign(x_image_undistorted) * np.sqrt(r2 * (x_image_undistorted**2) / (1 + x_image_undistorted**2 + z_image_undistorted**2))
  y_undistorted = np.sqrt(r2 * 1 / (1 + x_image_undistorted**2 + z_image_undistorted**2))
  z_undistorted = np.sign(z_image_undistorted) * np.sqrt(r2 * (z_image_undistorted**2) / (1 + x_image_undistorted**2 + z_image_undistorted**2))

  source_undistorted = np.stack([x_undistorted, y_undistorted, z_undistorted], 1)

  return source_undistorted

def apply_polynomial_distortion(source, w_x, w_z, scale, offset):
  """
  Given a set of distortion parameters, apply it to the point cloud
  """

  # Do un-distortion in image coordinate
  source_image, _ = to_image(source)

  x_image_undistorted = apply_polynomial(source_image[:, 0], source_image[:, 1], w_x).astype(np.float32)
  z_image_undistorted = apply_polynomial(source_image[:, 0], source_image[:, 1], w_z).astype(np.float32)

  # Transfrom from image coordinate to Cartesian coordinate
  r2 = (scale * np.linalg.norm(source.astype(np.float32), 2, 1) + offset)**2
  x_undistorted = np.sign(x_image_undistorted) * np.sqrt(r2 * (x_image_undistorted**2) / (1 + x_image_undistorted**2 + z_image_undistorted**2))
  y_undistorted = np.sqrt(r2 * 1 / (1 + x_image_undistorted**2 + z_image_undistorted**2))
  z_undistorted = np.sign(z_image_undistorted) * np.sqrt(r2 * (z_image_undistorted**2) / (1 + x_image_undistorted**2 + z_image_undistorted**2))

  source_undistorted = np.stack([x_undistorted, y_undistorted, z_undistorted], 1)

  return source_undistorted

def apply_BC_distortion(source, popt):
  """
  Brown–Conrady model
  """

  source_image, _ = to_image(source)

  undistorted_source_image = BC_model(source_image, *popt)
  undistorted_source_image = np.reshape(undistorted_source_image, np.shape(source_image))

  x_image_undistorted = undistorted_source_image[:, 0]
  z_image_undistorted = undistorted_source_image[:, 1]

  # Transfrom from image coordinate to Cartesian coordinate
  r2 = np.linalg.norm(source, 2, 1)**2
  x_undistorted = np.sign(x_image_undistorted) * np.sqrt(r2 * (x_image_undistorted**2) / (1 + x_image_undistorted**2 + z_image_undistorted**2))
  y_undistorted = np.sqrt(r2 * 1 / (1 + x_image_undistorted**2 + z_image_undistorted**2))
  z_undistorted = np.sign(z_image_undistorted) * np.sqrt(r2 * (z_image_undistorted**2) / (1 + x_image_undistorted**2 + z_image_undistorted**2))

  source_undistorted = np.stack([x_undistorted, y_undistorted, z_undistorted], 1)

  return source_undistorted

def to_angular(data, mode="rad"):
  """
  Transform from cartesian to angular. (Range info will be removed)
  The input data should be in the sensor reference frame.
  """
  d = np.linalg.norm(data, 2, 1)
  x = data[:, 0]
  y = data[:, 1]
  z = data[:, 2]

  ele = np.arctan2(z, y)
  azi = np.arctan2(x, y)

  if mode == "rad":
    return np.stack([azi, ele], 1)
  elif mode == "degree":
    return np.stack([azi, ele], 1) * 180 / np.pi
  else:
    raise ValueError("Unknown mode: {}".format(mode))
    
def to_image(data):
  """
  Transform from cartesian to image. (Range info will be removed)
  The input data should be in the sensor reference frame.
  """
  x = data[:, 0]
  y = data[:, 1]
  z = data[:, 2]

  x_ = x / y
  z_ = z / y

  return np.stack([x_, z_], 1), y

def distortion_model_x(x, z, w_x):
  """
  Not using more
  Azimuth:
    x' = x0 + (x  - x0)(1 + (a1*z + a0)(x - x0)**2 + (b1*z + b0)(x - x0)**4)
  """
  x0 = w_x[0]
  a0 = w_x[1]
  a1 = w_x[2]
  b0 = w_x[3]
  b1 = w_x[4]

  #return x0 + (x - x0) * (1 + (a0 + a1*z) * (x - x0)**2 + (b0 + b1*z) * (x - x0)**4)
  return x + (a0 + a1*z) * (x - x0)**3 + (b0 + b1*z) * (x - x0)**5

def distortion_model_z(x, z, w_z):
  """
  Not using anymore
  Elevation:
    1. z' = C0 + (1 + C1)*z + (d0 + d1*(x-x0)**2)*z**2 + (e0 + e1*(x-x0)**2)*z**3
  """
  z0 = w_z[0]
  a0 = w_z[1]
  a1 = w_z[2]
  a2 = w_z[3]
  b0 = w_z[4]
  b1 = w_z[5]
  b2 = w_z[6]

  return z + (a0 + a1*x + a2*x**2) * (z - z0)**3 + (b0 + b1*x + b2*x**2) * (z - z0)**5

def apply_polynomial(x, z, w):
  """
  Apply a nth order polynomial.
  result = w[0] + w[1]*x + w[2]*z + w[3]*x^2 + w[4]*xz + x[5]*z^2 + ...
  """
  res = np.zeros_like(x)

  assert NUM_DISTORTION_PARAM == len(w)
  
  param_idx = 0
  for i in range(POLYNOMIAL_ORDER+1):
    for z_order in range(i+1):
      x_order = i - z_order
      res += w[param_idx] * x**x_order * z**z_order
      param_idx += 1

  return res

def D_loss_L1(w, source, target, alpha):
  """
  Only correct distortion. 
  """
  x_src = source[:, 0]
  z_src = source[:, 1]

  x_tar = target[:, 0]
  z_tar = target[:, 1]

  w[1] += 1
  w[NUM_DISTORTION_PARAM+2] += 1

  w_x = w[:NUM_DISTORTION_PARAM]
  w_z = w[NUM_DISTORTION_PARAM:]

  x_pred = apply_polynomial(x_src, z_src, w_x)
  z_pred = apply_polynomial(x_src, z_src, w_z)

  loss_x = x_pred - x_tar
  loss_z = z_pred - z_tar

  weight = 100*(x_tar**2 + z_tar**2)
  weight /= weight.sum()

  distortion_nominal = np.zeros(2*NUM_DISTORTION_PARAM)
  distortion_nominal[1] = 1
  distortion_nominal[NUM_DISTORTION_PARAM+2] = 1

  abs_loss = np.sum(weight*(np.abs(loss_x)**2 + np.abs(loss_z)**2)) + \
             alpha * (np.sum((distortion_nominal - w)**2)) 

  return abs_loss

def RTD_loss_L1(w, source, target, alpha, trans, euler, apply_scale, apply_offset):
  """
  RTD means rotation, translation and distortion
  Compute the loss of current set of parameters
  """

  translation, rotation, distortion, w_x, w_z, scale, offset = \
      parameter_unpack(w, apply_scale, apply_offset)

  w_x[0] = 0
  w_z[0] = 0

  source_undistorted = apply_polynomial_distortion(source, w_x, w_z, scale, offset)

  R = transforms3d.euler.euler2mat(*rotation)
  source_undistorted_transformed = (R@source_undistorted.T + translation[:, None]).T

  distortion_nominal = np.zeros(2*NUM_DISTORTION_PARAM)
  distortion_nominal[1] = 1
  distortion_nominal[NUM_DISTORTION_PARAM+2] = 1

  abs_loss = np.mean(np.linalg.norm(source_undistorted_transformed - target, 2, 1)) + \
                     1000 * alpha * np.sum(np.abs(trans - translation)) + \
                     1000 * alpha * np.sum(np.abs(euler - rotation)) + \
                     alpha * np.sum(np.abs(distortion_nominal - distortion)**2)

  return abs_loss

def optimize_RTD(source, target, param, max_iter=500, tol=1e-5, alpha=1e-5, 
                 apply_scale=False, apply_offset=False, display=False):
  """
  Optimize Rotation, Translation and Distortion
  """
  n = len(source)

  parameters = param
  
  trans = parameters[:3]
  euler = parameters[3:6]

  bounds = np.tile([-np.inf, np.inf], (len(parameters), 1))
  bounds[3:6, 0] = -1
  bounds[3:6, 1] = 1

  opt_res = minimize(
      RTD_loss_L1, parameters, 
      args=(source, target, alpha, trans, euler, apply_scale, apply_offset),
      options={"maxiter": max_iter, "gtol": tol, "disp":display},
      bounds=bounds)
  parameters = opt_res.x

  return parameters

def optimize_D(source, target, max_iter=500, tol=1e-5, alpha=1e-5, 
                        param=None, display=False):
  """
  Optimize distortion
  """
  n = len(source)

  if param is None:
    parameters = np.zeros(NUM_DISTORTION_PARAM*2)
  else:
    parameters = param

  opt_res = minimize(
      D_loss_L1, parameters, 
      args=(source, target, alpha),
      options={"maxiter": max_iter, "gtol": tol, "disp":display})
  parameters = opt_res.x

  return parameters

def RT_loss_L1(w, source, target):

  translation = w[:3]
  rotation = w[3:]

  R = transforms3d.euler.euler2mat(*rotation)
  source_transformed = (R@source.T + translation[:, None]).T

  d = np.linalg.norm(source_transformed - target, 2, 1)

  return np.mean(d)

def optimize_RT(source, target, max_iter=500, tol=1e-5, 
                        param=None, display=False):
  """
  Optimize rotation and translation for rigid body transformation
  """
  n = len(source)

  if param is None:
    parameters = np.zeros(6)

  else:
    parameters = param

  bounds = np.zeros([6, 2])
  bounds[:3, 0] = -np.inf
  bounds[:3, 1] = np.inf
  bounds[3:6, 0] = -np.pi
  bounds[3:6, 1] = np.pi

  print(source.shape)
  print(target.shape)
  print(bounds.shape)

  opt_res = minimize(
      RT_loss_L1, parameters, 
      args=(source, target),
      options={"maxiter": max_iter, "gtol": tol, "disp":display},
      bounds=bounds)
  parameters = opt_res.x

  return parameters

def BC_model(x, xc, zc, k2, k3, k4, k5, k6, p1, p2, p3, p4):
  """
  Brown–Conrady model
  """
  xd = x[:,0] # distorted x
  zd = x[:,1]
  r = np.linalg.norm(np.stack((xd-xc, zd-zc)), axis = 0)
  dx = xd-xc
  dz = zd-zc
  # k5 = 0 # for test purpose
  xu = xd + (xd-xc)*(k2*r**2+k3*r**3+k4*r**4+k5*r**5+k6*r**6) + \
      (p1*(r**2 + 2*dx**2) + 2*p2*dx*dz)*(1+p3*r**2+p4*r**4)
  zu = zd + (zd-zc)*(k2*r**2+k3*r**3+k4*r**4+k5*r**5+k6*r**6) + \
      (2*p2*dx*dz + p2*(r**2+2*dz**2))*(1+p3*r**2+p4*r**4)
  A = np.stack([xu, zu], 1)
  A = A.flatten()
  return A    

def optimize_BC_model(source, target):
  """
  Brown–Conrady model
  """
  from scipy.optimize import curve_fit
  c0 = (0, 0, 1e-3, -5e-1, 5e-3, 5e-3, 1e-6, 1e-6, 1e-6, 1e-3, 1e-3) # for model 2

  popt, _ = curve_fit(BC_model, source, target.flatten(), p0 = c0)

  return popt
