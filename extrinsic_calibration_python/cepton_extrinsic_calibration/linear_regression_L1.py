###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

from math import inf
import numpy as np
from scipy.optimize import minimize
import matplotlib.path as mplPath

def L1_loss_and_grad(w, X, y, outer_scale):
  n, n_feature = X.shape
  assert(len(w) == n_feature)

  prediction = np.inner(w, X) + 1
  linear_loss = prediction - y

  inner_mask = prediction > 0
  outer_mask = ~inner_mask

  abs_linear_loss = np.abs(linear_loss) / n

  abs_linear_loss[outer_mask] *= outer_scale

  grad = np.zeros(n_feature)

  signed_grad = np.zeros_like(linear_loss)
  positive_mask = linear_loss > 0
  negative_mask = ~positive_mask

  signed_grad[positive_mask] = 1.
  signed_grad[negative_mask] = -1.

  signed_grad[outer_mask] *= outer_scale
  
  grad[:n_feature] = np.dot(signed_grad[None, :], X) / n

  return np.sum(abs_linear_loss), grad

def L1_regression(X, y, outer_scale, max_iter = 100, tol=1e-05):
  n, n_feature = X.shape
  parameters = np.zeros(n_feature)

  bounds = np.tile([-np.inf, np.inf], (parameters.shape[0], 1))

  opt_res = minimize(
    L1_loss_and_grad, parameters, method="L-BFGS-B", jac=True,
    args=(X, y, outer_scale),
    options={"maxiter": max_iter, "gtol": tol, "iprint": -1},
    bounds=bounds)
  
  parameters = opt_res.x

  return parameters

def bounding_box_outlier_rate(w, X, dist):
  """
  """
  offset = w[:2] * 1e6
  theta = w[2] * 1e6

  width = w[3] * 1e6
  height = w[4] * 1e6

  R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
  ])

  poly_vertice = np.array([[0, 0], [0, height], [width, height], [width, 0]])

  rot_poly_vertice = (R @ poly_vertice.T).T

  rot_poly_vertice += offset[None, :]

  poly_path = mplPath.Path(rot_poly_vertice)
  in_mask = poly_path.contains_points(X)

  error = 1 - np.mean(in_mask)
  return error  + np.abs(1 - width * height / (dist[0] * dist[1])) / 10

def bounding_box_regression(X, dist, max_iter=100, tol=1e-5):
  init_parameters = np.zeros(5)

  init_parameters[:2] = (np.mean(X, 0) - dist / 2) / 1e6

  init_parameters[3:] = dist / 1e6

  bounds = np.array([
    [-np.inf, np.inf],
    [-np.inf, np.inf],
    [-np.pi/4, np.pi/4],
    [dist[0] / 1e6, (dist[0] + 0.1) / 1e6],
    [dist[1] / 1e6, (dist[1] + 0.1) / 1e6]
  ])

  opt_res = minimize(
    bounding_box_outlier_rate, init_parameters,
    args=(X, dist),
    options={"maxiter": max_iter, "gtol": tol, "disp":False},
    bounds=bounds)
  
  parameters = opt_res.x


  offset = parameters[:2] * 1e6
  theta = parameters[2] * 1e6

  width = parameters[3] * 1e6
  height = parameters[4] * 1e6

  R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
  ])

  poly_vertice = np.array([[0, 0], [0, height], [width, height], [width, 0]])

  rot_poly_vertice = (R @ poly_vertice.T).T

  rot_poly_vertice += offset[None, :]

  return parameters[2] * 1e6 

if __name__ == "__main__":

  xx, yy = np.meshgrid(np.arange(-3 ,3), np.arange(-3, 3))
  x_flatten = xx.flatten()
  y_flatten = yy.flatten()

  X = np.stack([x_flatten, y_flatten], 1)
  zz = x_flatten*3 + y_flatten*4 + 5

  w = np.array([1, 2, 1])
  #loss, grad = L1_loss_and_grad(w, X, y, 0.5)

  param = L1_regression(X, zz, 0.5)

  print(param)
