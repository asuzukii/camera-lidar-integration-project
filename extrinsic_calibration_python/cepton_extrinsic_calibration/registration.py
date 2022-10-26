###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

from .models.distortion import *
from .models.bezier import BezierModel
from .plots import plot_3D, parameter_distortion_map
from .data_io import *
from .paramslib import parameter_NAL

def kabsch_umeyama(A, B, w=None):
  """
  Kabsch Umeyama method
  Calculates the optimal rotation matrix
  """
  assert A.shape == B.shape
  n, m = A.shape

  if not w is None:
    assert len(w) == n
    w /= np.sum(w)
  else:
    w = np.ones(n)/n

  EA = np.mean(A, axis=0)
  EB = np.mean(B, axis=0)
  #VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

  H = ((A - EA).T @ np.diag(w) @ (B - EB)) / n
  U, D, VT = np.linalg.svd(H)
  d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
  S = np.diag([1] * (m - 1) + [d])

  R = U @ S @ VT
  c = 1 #VarA / np.trace(np.diag(D) @ S)
  t = EA - c * R @ EB

  return R, t

def registration(source, target, num_per_chessboard=None, display=False):
  """
  Register the source and target with:
  1. Bezier distortion correction model applied on source
  2. Affine 3D matrix applied on source (after correction of step 1)
  """
  
  model = BezierModel()
  model.load_parameter(parameter_NAL["distortion"])

  source_pre = apply_bezier_distortion(source, model)

  if display:
    plot_3D(source_pre, target, "input (initial distortion parameters)")

  if not num_per_chessboard is None:
    source_per_target = []
    curr_idx = 0
    for num_corners in num_per_chessboard:
      source_per_target.append(source[curr_idx: curr_idx+num_corners])
      curr_idx += num_corners

  weight = np.sqrt(source_pre[:, 1]**2 / (source_pre[:, 0]**2 + source_pre[:, 2]**2))
  weight /= weight.sum()

  R, t = kabsch_umeyama(target, source_pre, None)

  transformed_source = (R @ source.T + t[:, None]).T

  transform_matrix = np.zeros([4, 4])
  transform_matrix[:3, :3] = R
  transform_matrix[:3, 3] = t
  transform_matrix[3, 3] = 1

  # Print this way so that it's easier to copy and paste to CloudCompare
  print("transformation matrix:")
  for i in range(4):
    print(transform_matrix[i, 0], transform_matrix[i, 1], transform_matrix[i, 2], transform_matrix[i, 3])

  if not num_per_chessboard is None:
    curr_index = 0
    for chessboard_idx, num_corners in enumerate(num_per_chessboard):
      target_errors = np.linalg.norm((transformed_source[curr_index:curr_index+num_corners] - \
          target[curr_index:curr_index+num_corners]), 2, 1)
      curr_index += num_corners

      print("chessboard {}: avg: {}, max: {}".format(chessboard_idx, target_errors.mean(), target_errors.max()))

  if display:
    plot_3D(target, transformed_source, "extrinsic transformation applied")

  fine_tune_transform = np.eye(4)
  fine_tune_transform[:3, :3] = R
  fine_tune_transform[:3, 3] = t

  parameters = {
    "transformation": fine_tune_transform,
    "distortion": np.array(model.parameters)
  }

  return parameters

def evaluate_parameters(source, target, corner_lengths, parameters, save_dir):
  """
  Evaluate the performance of given parameters
  """

  transform_matrix = parameters["transformation"]
  distortion_parameters = parameters["distortion"]

  model = BezierModel()
  model.load_parameter(distortion_parameters)

  R = transform_matrix[:3, :3]
  translation = transform_matrix[:3, 3]

  source_undistorted = apply_bezier_distortion(source, model)

  source_undistorted_transformed = (R@source_undistorted.T + translation[:, None]).T

  csv_writer(os.path.join(save_dir, "undistorted_corner.csv"), source_undistorted_transformed)

  transform_matrix = np.eye(4)
  transform_matrix[:3, :3] = R
  transform_matrix[:3, 3] = translation

  transform_matrix_inv = np.linalg.inv(transform_matrix)
  target_in_Lidar_frame = (transform_matrix_inv[:3, :3] @ target.T + transform_matrix_inv[:3, 3, None]).T

  csv_writer(os.path.join(save_dir, "target_in_Lidar_frame.csv"), target_in_Lidar_frame)
  csv_writer(os.path.join(save_dir, "source_undistorted.csv"), source_undistorted)

  source_angular = to_angular(source)
  target_angular = to_angular(target_in_Lidar_frame)
  source_undistorted_angular = to_angular(source_undistorted)

  print("===========> range error analysis <===========")
  curr_index = 0
  for chessboard_idx, num_corners in enumerate(corner_lengths):
    range_error = np.mean((source_undistorted_transformed[curr_index:curr_index+num_corners] - \
                           target[curr_index:curr_index+num_corners]), 0)
    print("range error {}: {}".format(chessboard_idx, range_error))
    curr_index += num_corners

  print("===========> cartesian error analysis <===========")
  curr_index = 0
  for chessboard_idx, num_corners in enumerate(corner_lengths):
    target_errors = np.linalg.norm((source_undistorted_transformed[curr_index:curr_index+num_corners] - \
        target[curr_index:curr_index+num_corners]), 2, 1)
    curr_index += num_corners

    print("chessboard {}: avg: {}, max: {}".format(chessboard_idx, target_errors.mean(), target_errors.max()))

  # angular error analysis
  print("===========> angular error analysis <===========")
  max_angular_error = np.max(np.abs(target_angular - source_angular), 0)
  mean_angular_error = np.mean(np.abs(target_angular - source_angular), 0)

  print("before correction maximum angular x: {} deg".format(max_angular_error[0] * 180. / np.pi))
  print("before correction maximum angular z: {} deg".format(max_angular_error[1] * 180. / np.pi))
  print("before correction mean angular x: {} deg".format(mean_angular_error[0] * 180. / np.pi))
  print("before correction mean angular z: {} deg".format(mean_angular_error[1] * 180. / np.pi))

  max_angular_error = np.max(np.abs(target_angular - source_undistorted_angular), 0)
  mean_angular_error = np.mean(np.abs(target_angular - source_undistorted_angular), 0)

  print("after correction maximum angular x: {} deg".format(max_angular_error[0] * 180. / np.pi))
  print("after correction maximum angular z: {} deg".format(max_angular_error[1] * 180. / np.pi))
  print("after correction mean angular x: {} deg".format(mean_angular_error[0] * 180. / np.pi))
  print("after correction mean angular z: {} deg".format(mean_angular_error[1] * 180. / np.pi))

  print("===========> parameters <===========")
  print(translation)
  print(R)
  print(distortion_parameters)

  ##### 3D image #####
  plot_3D(target, source_undistorted_transformed, "extrinsic+distortion applied")
  
  target_angular_deg = target_angular * 180 / np.pi
  source_angular_deg = source_angular * 180 / np.pi
  source_undistorted_angular_deg = source_undistorted_angular * 180 / np.pi

  ##### w/o correction #####
  print("===========> w/o correction <===========")
  plt.figure(figsize=(20,10))
  plt.subplot(211)
  plt.scatter(target_angular_deg[:, 0], target_angular_deg[:, 1], c='b')
  plt.scatter(source_angular_deg[:, 0], source_angular_deg[:, 1], c='r')

  error_in = source_angular_deg - target_angular_deg
  error_dist = np.linalg.norm(error_in, 2, 1)
  error_rank = np.argsort(error_dist)[::-1]
  
  for i in range(30):
    plt.text(target_angular_deg[error_rank[i], 0], 
             target_angular_deg[error_rank[i], 1], str(i+1))

    print("idx: {}/{}, ({:.2f}, {:.2f})".format(i+1, len(error_in), 
                                                error_in[error_rank[i], 0], 
                                                error_in[error_rank[i], 1]))

  plt.title("W/o correction")
  plt.legend(["gd corners", "scan corners"])
  plt.ylabel("elevation (deg)")

  ##### w/ correction #####
  print("===========> w/ correction <===========")
  plt.subplot(212)
  plt.scatter(target_angular_deg[:, 0], target_angular_deg[:, 1], c='b')
  plt.scatter(source_undistorted_angular_deg[:, 0], source_undistorted_angular_deg[:, 1], c='r')

  error_out = source_undistorted_angular_deg - target_angular_deg
  error_dist = np.linalg.norm(error_out, 2, 1)
  error_rank = np.argsort(error_dist)[::-1]

  for i in range(30):
    plt.text(target_angular_deg[error_rank[i], 0], 
             target_angular_deg[error_rank[i], 1], str(i+1))
    print("idx: {}/{}, ({:.2f}, {:.2f})".format(i+1, len(error_out), 
                                                error_out[error_rank[i], 0], 
                                                error_out[error_rank[i], 1]))

  plt.title("W/ correction")
  plt.legend(["gd corners", "undistorted scan corners"])
  plt.ylabel("elevation (deg)")
  
  parameter_distortion_map(distortion_parameters, save_dir)

  plt.figure(figsize=(20, 8))
  plt.subplot(121)
  plt.scatter(target_angular_deg[:, 0], source_angular_deg[:, 0] - target_angular_deg[:, 0], marker='+', s=50, c='b')
  plt.title('x error - w/o correction')
  plt.xlabel("x field angle [deg]")
  plt.ylabel("x angular error [deg]")
  plt.grid(True, axis="y")

  plt.subplot(122)
  plt.scatter(target_angular_deg[:, 1], source_angular_deg[:, 1] - target_angular_deg[:, 1], marker='+', s=50, c='b')
  plt.title('z error - w/o correction')
  plt.xlabel("z field angle [deg]")
  plt.ylabel("z angular error [deg]")
  plt.grid(True, axis="y")

  return error_in, error_out, target_angular_deg

def error_analysis(error_in, error_out, ref_angular, valid_corners_index, 
                   corner_lengths, layouts, save_dir, mode):
  """
  error analysis
  """
  valid_corners_index_list = []
  batch_offset = 0
  for length in corner_lengths:
    valid_corners_index_list.append(valid_corners_index[batch_offset:batch_offset+length])
    batch_offset += length

  batch_offset = 0

  plt.figure(figsize=(20, 8))
  ax_x = plt.subplot(121)
  ax_z = plt.subplot(122)

  error_out_x_all = []
  error_out_z_all = []
  error_in_x_all = []
  error_in_z_all = []

  angular_x_all = []
  angular_z_all = []

  for layout, valid_corners in zip(layouts, valid_corners_index_list):
    num_ver, num_hor = layout
    
    angular_x = []
    angular_z = []

    error_in_x = []
    error_in_z = []
    error_out_x = []
    error_out_z = []

    for row in range(num_ver-1):
      for col in range(num_hor-1):

        neighbor_error_in = []
        neighbor_error_out = []
        ref_angular_position = []
        # try to get the sum of nearby 4 corners (itself included)
        for offset_row, offset_col in [[0,0], [0,1], [1,0], [1,1]]:
          row_idx = row + offset_row
          col_idx = col + offset_col
          idx_in_seq = np.where(valid_corners == row_idx*num_hor + col_idx)

          if idx_in_seq[0].size > 0:
            neighbor_error_in.append(error_in[idx_in_seq[0] + batch_offset])
            neighbor_error_out.append(error_out[idx_in_seq[0] + batch_offset])

            ref_angular_position.append(ref_angular[idx_in_seq[0] + batch_offset,:])

        if len(neighbor_error_in) > 2:

          avg_error_in = np.mean(np.array(neighbor_error_in), 0)
          avg_error_out = np.mean(np.array(neighbor_error_out), 0)
          avg_angular = np.mean(np.array(ref_angular_position), 0)

          angular_x.append(avg_angular[0, 0])
          angular_z.append(avg_angular[0, 1])

          error_in_x.append(avg_error_in[0, 0])
          error_in_z.append(avg_error_in[0, 1])

          error_out_x.append(avg_error_out[0, 0])
          error_out_z.append(avg_error_out[0, 1])
    
    error_out_x_all += error_out_x
    error_out_z_all += error_out_z
    error_in_x_all += error_in_x
    error_in_z_all += error_in_z
    angular_x_all += angular_x
    angular_z_all += angular_z

    batch_offset += len(valid_corners)

  np.save(os.path.join(save_dir, "{}_angular_x.npy".format(mode)), np.array(angular_x_all))
  np.save(os.path.join(save_dir, "{}_angular_z.npy".format(mode)), np.array(angular_z_all))

  np.save(os.path.join(save_dir, "{}_error_out_x_all.npy".format(mode)), np.array(error_out_x_all))
  np.save(os.path.join(save_dir, "{}_error_out_z_all.npy".format(mode)), np.array(error_out_z_all))

  np.save(os.path.join(save_dir, "{}_error_in_x_all.npy".format(mode)), np.array(error_in_x_all))
  np.save(os.path.join(save_dir, "{}_error_in_z_all.npy".format(mode)), np.array(error_in_z_all))

  #ax_x.scatter(angular_x_all, error_in_x_all, color='r', marker='x')
  ax_x.scatter(angular_x_all, error_out_x_all, color='b', marker='+', s=50)

  #ax_z.scatter(angular_z_all, error_in_z_all, color='r', marker='x')
  ax_z.scatter(angular_z_all, error_out_z_all, color='b', marker='+', s=50)

  ax_x.set_title("{} x error - w/ correction".format(mode))
  ax_x.set_xlabel("x field angle [deg]")
  ax_x.set_ylabel("x angular error [deg]")
  ax_x.grid(True, axis="y")

  ax_z.set_title("{} z error - w/ correction".format(mode))
  ax_z.set_xlabel("z field angle [deg]")
  ax_z.set_ylabel("z angular error [deg]")
  ax_z.grid(True, axis="y")


  plt.savefig(os.path.join(save_dir, "{} error plot.png".format(mode)))
