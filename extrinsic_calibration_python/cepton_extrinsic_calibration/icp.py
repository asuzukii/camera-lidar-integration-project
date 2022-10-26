###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
from .registration import kabsch_umeyama

def ICP_correspondence(target, source, pixel_per_square):
  """
  Use ICP to find the corresponding corners
  """
  threshold = pixel_per_square / 2
  # Initialize
  t = np.array([0, 0])
  R = np.ones([2,2])
  iter = 0
  source_copy = source - np.mean(source, 0) + np.mean(target, 0)

  while iter == 0 or np.linalg.norm(t) > 1e-5 or np.arccos(np.clip(R[0, 0], -1, 1)) > 1e-5:
    
    dist_mat = np.linalg.norm(source_copy[:, None, :] - target[None, :, :], 2, 2)

    nn = np.argmin(dist_mat, 1)
    R, t = kabsch_umeyama(target[nn, :], source_copy)

    source_copy = (R @ source_copy.T + t[:, None]).T

    iter += 1

  dist_mat = np.linalg.norm(source_copy[:, None, :] - target[None, :, :], 2, 2)
  # Find the nearest neighbor for each groundtruth target pattern
  nn = np.argmin(dist_mat, 0)
  minimum_dist = np.min(dist_mat, 0)
  corrsponding_detected_index = nn[minimum_dist < threshold]
  valid_target_index = np.arange(len(target))[minimum_dist < threshold]    
  valid_count = np.sum(minimum_dist < threshold)

  for offset in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
    target_offset = target + np.array(offset)[None, :]*pixel_per_square
    dist_mat = np.linalg.norm(source_copy[:, None, :] - target_offset[None, :, :], 2, 2)
    # Find the nearest neighbor for each groundtruth target pattern
    nn = np.argmin(dist_mat, 0)
    minimum_dist = np.min(dist_mat, 0)

    if valid_count < np.sum(minimum_dist < threshold):
      valid_count = np.sum(minimum_dist < threshold)
      corrsponding_detected_index = nn[minimum_dist < threshold]
      valid_target_index = np.arange(len(target))[minimum_dist < threshold]    
      print("better: ", offset)

  return valid_target_index, corrsponding_detected_index
