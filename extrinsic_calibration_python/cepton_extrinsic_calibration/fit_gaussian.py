###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.sparse import coo

def gaussian_2D(x, x0, y0, xsigma, ysigma, A):
    return A * np.exp( -((x[:, 0]-x0)/xsigma)**2 -((x[:, 1]-y0)/ysigma)**2)

def fit_gaussian_2D(coords, density):
  
  n, dim = coords.shape
  assert dim == 2

  max_idx = np.argmax(density)

  try:
    p = (coords[max_idx, 0], coords[max_idx, 1], max(np.std(coords[:, 0]), 1e-5), max(np.std(coords[:, 1]), 1e-5), density[max_idx])
    parameters, _ = curve_fit(gaussian_2D, coords, density, p0=p)
    return parameters[:2], parameters[4]

  except:
    return coords[max_idx, :], density[max_idx]
