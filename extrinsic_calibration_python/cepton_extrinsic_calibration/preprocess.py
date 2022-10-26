###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
import matplotlib.pyplot as plt

#from .modules.c_preprocess import PreprocessCpp
from .registration import kabsch_umeyama

# def preprocess(cloud: np.ndarray, x_min, x_max, y_min, y_max, z_min, z_max, 
#                intensity_min, intensity_max):
#   output = PreprocessCpp(cloud, x_min, x_max, y_min, y_max, z_min, z_max, 
#                intensity_min, intensity_max)
#   data = output[:, :3]

#   w = np.linalg.inv(data.T @ data) @ data.T @ np.ones([len(data), 1])

#   return output

def preprocess(cloud: np.ndarray, x_min, x_max, y_min, y_max, z_min, z_max, 
               intensity_min, intensity_max):
  """
  Preprocess:
    Cropping: Discard any point that isn't in the ROI.
    Info filtering: We need x, y, z, intensity, segment_id (optional)
    Probably add more in the future
  Input:
    cloud: np.ndarray of points
    ROI: minimum and maximum value for x, y, z and intensity (inclusive)
  Return: an array of point cloud, n * 4
  """

  assert len(cloud) > 0

  mask = cloud[:, 4].astype(np.bool8) & np.logical_not(cloud[:, 5].astype(np.bool8)) & \
            (cloud[:, 0] >= x_min) & (cloud[:, 0] <= x_max) & \
            (cloud[:, 1] >= y_min) & (cloud[:, 1] <= y_max) & \
            (cloud[:, 2] >= z_min) & (cloud[:, 2] <= z_max) & \
            (cloud[:, 3] >= intensity_min) & (cloud[:, 3] <= intensity_max)
  mask = mask.astype(bool)
  filter_data = np.delete(cloud[mask], [4,5], 1)
  # Only normalize if we have channel info
  print("Normalizing channels...") 
  filter_data = normalize_channel(filter_data, 0.1, 0.5, show_intensities=True)

  # Remove redundant point
  print("Remove redundant points...") 
  point_cloud_simplified = remove_redundant(filter_data)

  return point_cloud_simplified

def normalize_channel(cloud: list, black_intensity=0.1, white_intensity=0.5,
                      discard_sampling = 0.05, show_intensities=False,
                      use_new_normalizer=True, mid_band_width=0.5):

  cloud_arr = np.array(cloud)
  #print(cloud_arr.shape)

  for i in range(64):
    channel_mask = cloud_arr[:, 4] == i
    channel_arr = cloud_arr[channel_mask]
    
    if len(channel_arr) < 5000:
      cloud_arr[channel_mask, 3] = np.clip(cloud_arr[channel_mask, 3], black_intensity, white_intensity)
      continue
    
    r_arr = channel_arr[:,3]

    first = np.percentile(r_arr, 75)
    second = np.percentile(r_arr, 25)

    alpha = (white_intensity - black_intensity) / (first - second + 1e-5)

    LUT = np.arange(0, 1.5+1e-5, 0.01)
    LUT = np.clip((LUT - second) * alpha + black_intensity, black_intensity, white_intensity)

    '''plt.figure()
    plt.plot(LUT)

    print(first, second, len(channel_arr))
    plt.figure()
    den, edges = np.histogram(r_arr, density=False, bins=np.linspace(0,1.0,num=101))
    plt.plot(edges[:-1], den)
    plt.show()'''

    cloud_arr[channel_mask, 3] = LUT[np.round((cloud_arr[channel_mask, 3] / 0.01).astype(float)).astype(np.int32)]

  return cloud_arr

def remove_redundant(point_cloud):
  """
  Average the intensity of points with the same x, y, z. Reduce redundant points 
  to one single point, with the average intensity.  
  Input:
    point_cloud: input point cloud
  Return:
    A point cloud that have no redundant point
  """
  grid = {}

  for point in point_cloud:
    # Use a string of coordinate as the key
    coord_key = (point[0], point[1], point[2])
    if not coord_key in grid.keys():
      grid[coord_key] = [point[3]]
    else:
      grid[coord_key].append(point[3])

  point_cloud_simplify = []

  for key in grid:
    intensities = grid[key]
    point_average = np.array([key[0], key[1], key[2], np.mean(intensities)])
    point_cloud_simplify.append(point_average)

  point_cloud_simplify = np.array(point_cloud_simplify)
  return point_cloud_simplify

def compute_each_corner(chessboard_spec):
  """
  Given measurement and the position in target frame, we can transform all the 
  corners to the measurement frame.
  Input:
    chessboard_spec[list]: list of dict
  """

  for chessboard in chessboard_spec:
    x_tick = np.array(chessboard["x_tick"])
    z_tick = np.array(chessboard["z_tick"])

    meas_marker = np.array(chessboard["measurement"]["markers"])
    correspondence = np.array(chessboard["measurement"]["correspondence"])

    # Convert from mm to m
    if np.mean(meas_marker) > 100:
      meas_marker /= 1000

    design_marker = np.stack([x_tick[correspondence[:, 0]], np.zeros(len(meas_marker)), z_tick[correspondence[:, 1]]], 1)

    R, t = kabsch_umeyama(meas_marker, design_marker)

    error_3d = (R@ design_marker.T + t[:, None]).T - meas_marker
    error_dist = np.linalg.norm(error_3d, 2, 1)
    
    if np.max(error_dist) > 0.01:
      print("Corners error > 1cm. Please check config json file")

    inner_corners = []
    # Descending on z axis
    for z in z_tick[1:-1][::-1]:
      for x in x_tick[1:-1]:
        inner_corners.append([x, 0, z])
    inner_corners = np.array(inner_corners)

    corners_gd = (R@ inner_corners.T + t[:, None]).T

    chessboard["corners"] = corners_gd
