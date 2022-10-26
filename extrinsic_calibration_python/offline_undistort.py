"""
Temporary script to estimate big wall distortion parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cepton_extrinsic_calibration.registration import *
from cepton_extrinsic_calibration.models.bezier import *
from cepton_extrinsic_calibration.models.distortion import *
#from _long_range_utils import *
from cepton_extrinsic_calibration.paramslib import *
from sklearn import linear_model
from cepton_extrinsic_calibration.data_io import *
from scipy import interpolate

plt.rcParams.update({'font.size': 20})

def undistort(corner_wo, corner_w, name, save_dir = None, return_plot_handle=False):

  '''if not os.path.exists(os.path.join(save_dir, "distortion_map")):
      os.mkdir(os.path.join(save_dir, "distortion_map"))
  if not os.path.exists(os.path.join(save_dir, "error_plot")):
      os.mkdir(os.path.join(save_dir, "error_plot"))'''
  target_image, _ = to_image(corner_wo)
  source_image, _ = to_image(corner_w)

  # The affine3d was applied, start initial distortion optimization
  model = BezierModel()

  #model.load_input(source_image[mask, :], target_image[mask, :])
  model.load_input(source_image, target_image)
  model.optimize_bazier()
  print(model.parameters)
  
  parameters = {
      "transformation": np.eye(4),
      "distortion": model.parameters
  }

  transform_matrix = parameters["transformation"]
  distortion_parameters = parameters["distortion"]

  print(parameters)
  x_map, z_map = model.draw_distortion_map()
  #np.save("./output/x_map_{}".format(name), x_map)
  #np.save("./output/z_map_{}".format(name), z_map)

  source_undistorted = apply_bezier_distortion(corner_w, model)

  source_angular = to_angular(corner_w)
  target_angular = to_angular(corner_wo)
  undistort_angular = to_angular(source_undistorted)

  source_angular_degree = source_angular * 180 / np.pi
  target_angular_degree = target_angular * 180 / np.pi
  undistort_angular_degree = undistort_angular * 180 / np.pi

  error_deg = (source_angular - target_angular) * 180 / np.pi

  x_target = target_angular[:, 0] * 180 / np.pi
  z_target = target_angular[:, 1] * 180 / np.pi

  reg_x = linear_model.LinearRegression()
  X = x_target[:, None]
  reg_x.fit(X, error_deg[:, 0])
  a_x = reg_x.coef_
  b_x = reg_x.intercept_

  reg_z = linear_model.LinearRegression()
  Z = z_target[:, None]
  reg_z.fit(Z, error_deg[:, 1])
  a_z = reg_z.coef_
  b_z = reg_z.intercept_

  fig_x = plt.figure(figsize=(10, 10))
  ax_x = fig_x.add_subplot(111)
  ax_x.scatter(target_angular[:, 0] * 180 / np.pi, error_deg[:, 0], marker='x', s=50, label="Measured WS distortion")
  #x_grid = np.arange(x_target.min(), x_target.max())
  #plt.plot(x_grid, a_x * x_grid + b_x)
  ax_x.set_title("x angular error - (w/ - w/o)")
  ax_x.set_xlabel("x field angle [deg]")
  ax_x.set_ylabel("x angular error [deg]")
  ax_x.grid()
  #plt.suptitle(name)
  #fig_x.savefig(os.path.join(save_dir, "error_plot", "corner_error_x_{}.png".format(name)))

  fig_z = plt.figure(figsize=(10, 10))
  ax_z = fig_z.add_subplot(111)
  ax_z.scatter(target_angular[:, 1] * 180 / np.pi, error_deg[:, 1], marker='x', s=50, label="Measured WS distortion")
  #z_grid = np.arange(z_target.min(), z_target.max())
  #plt.plot(z_grid, a_z * z_grid + b_z)
  ax_z.set_title("z angular error - (w/ - w/o)")
  ax_z.set_xlabel("z field angle [deg]")
  ax_z.set_ylabel("z angular error [deg]")
  ax_z.grid()
  #plt.suptitle(name)
  #fig_z.savefig(os.path.join(save_dir, "error_plot", "corner_error_z_{}.png".format(name)))

  plt.figure(figsize=(10,10))
  plt.scatter(undistort_angular_degree[:,0 ], undistort_angular_degree[:, 1], marker='x', label="undistort")
  plt.scatter(target_angular_degree[:,0 ], target_angular_degree[:, 1], marker='x', label='target')
  plt.scatter(source_angular_degree[:,0 ], source_angular_degree[:, 1], marker='x', label='source')
  plt.xlabel("x field angle [deg]")
  plt.ylabel("z field angle [deg]")
  plt.grid()
  plt.legend()

  residual_error = target_angular_degree - undistort_angular_degree
  
  x_margin = np.std(residual_error[:, 0]) * 3
  z_margin = np.std(residual_error[:, 1]) * 3

  mask = (np.abs(residual_error[:, 0]) < x_margin) & (np.abs(residual_error[:, 1]) < z_margin)

  plt.figure(figsize=(22,10))
  plt.subplot(121)
  plt.scatter(target_angular_degree[mask, 0], residual_error[mask, 0], marker='x', label='x angular error')
  plt.xlabel("x field angle [deg]")
  plt.ylabel("x angular error [deg]")
  plt.title("peak to peak error: +/-{:.3f} deg".format(3 * np.std(residual_error[mask, 0])))
  plt.grid()
  plt.legend()

  plt.subplot(122)
  plt.scatter(target_angular_degree[mask, 0], residual_error[mask, 1], marker='x', label='z angular error')
  plt.xlabel("x field angle [deg]")
  plt.ylabel("z angular error [deg]")
  plt.title("peak to peak error: +/-{:.3f} deg".format(3 * np.std(residual_error[mask, 1])))
  plt.grid()
  plt.legend()


  if return_plot_handle:
      return a_x, b_x, a_z, b_z, ax_x, ax_z
  else:
      return a_x, b_x, a_z, b_z
    
def estimate_distortion(corners_gd, corners_ws, npy_name: str):

    corners_gd_x, corners_gd_z = np.arctan2(corners_gd[:, 0], corners_gd[:, 1]), np.arctan2(corners_gd[:, 2], corners_gd[:, 1])
    corners_ws_x, corners_ws_z = np.arctan2(corners_ws[:, 0], corners_ws[:, 1]), np.arctan2(corners_ws[:, 2], corners_ws[:, 1])

    corners_gd_angular = np.stack([corners_gd_x, corners_gd_z], 1)*180/np.pi
    corners_ws_angular = np.stack([corners_ws_x, corners_ws_z], 1)*180/np.pi

    diff = np.linalg.norm(corners_gd_angular[None, :, :3] - corners_ws_angular[:, None, :3], 2, 2)

    valid_pair_1 = (np.min(diff, 0) < 0.5).nonzero()[0]
    valid_pair_2 = diff.argmin(0)[valid_pair_1]

    plt.figure(figsize=(10, 10))
    plt.scatter(corners_gd_x* 180 / np.pi, corners_gd_z* 180 / np.pi)
    plt.scatter(corners_ws_x* 180 / np.pi, corners_ws_z* 180 / np.pi)

    corners_gd = corners_gd[valid_pair_1]
    corners_ws = corners_ws[valid_pair_2]

    corners_gd_x, corners_gd_z = np.arctan2(corners_gd[:, 0], corners_gd[:, 1])*180/np.pi, np.arctan2(corners_gd[:, 2], corners_gd[:, 1])*180/np.pi
    corners_ws_x, corners_ws_z = np.arctan2(corners_ws[:, 0], corners_ws[:, 1])*180/np.pi, np.arctan2(corners_ws[:, 2], corners_ws[:, 1])*180/np.pi

    corners_gd = corners_gd[(corners_gd_z < 11) & (corners_gd_z > -10)] 
    corners_ws = corners_ws[(corners_gd_z < 11) & (corners_gd_z > -10)]

    corners_ws_x_deg = corners_ws_x * 180 / np.pi
    corners_ws_z_deg = corners_ws_z * 180 / np.pi
    corners_gd_x_deg = corners_gd_x * 180 / np.pi
    corners_gd_z_deg = corners_gd_z * 180 / np.pi

    fieldAngleZLower = corners_gd_z_deg.min()
    fieldAngleZHigher = corners_gd_z_deg.max()

    print(fieldAngleZLower, fieldAngleZHigher)


    corners_gd_x, corners_gd_z = np.arctan2(corners_gd[:, 0], corners_gd[:, 1]), np.arctan2(corners_gd[:, 2], corners_gd[:, 1])
    corners_ws_x, corners_ws_z = np.arctan2(corners_ws[:, 0], corners_ws[:, 1]), np.arctan2(corners_ws[:, 2], corners_ws[:, 1])

    corners_ws_x_deg = corners_ws_x * 180 / np.pi
    corners_ws_z_deg = corners_ws_z * 180 / np.pi
    corners_gd_x_deg = corners_gd_x * 180 / np.pi
    corners_gd_z_deg = corners_gd_z * 180 / np.pi

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.scatter(corners_gd_x_deg, corners_ws_x_deg - corners_gd_x_deg, marker='x', c=corners_gd_z_deg)
    ax.set_title("X Angular error along x axis")
    ax.set_xlabel("X Field Angle [deg]")
    ax.set_ylabel("X Anglular Error [deg]")
    ax.grid()
    fig.colorbar(im, label='Z Field Angle [deg]')

    plt.figure(figsize=(10, 10))
    plt.scatter(corners_gd_x_deg, corners_ws_z_deg - corners_gd_z_deg, marker='x', c=corners_gd_z_deg)
    plt.colorbar(label='Z Field Angle [deg]')
    plt.title("Z Angular error along x axis")
    plt.xlabel("X Field Angle [deg]")
    plt.ylabel("Z Anglular Error [deg]")
    plt.grid()

    undistort(corners_gd, corners_ws, npy_name, "./output/")

    plt.show()

"""NAL data"""

wsIndices = ["162132", "162844", "162916", "162932", "162951", "163007", "163120", "163206", "163225", "163350"]
#wsIndices = ["163120", "163206"]

corners_gd_data = []
corners_ws_data = []
for wsIdx in wsIndices:
    #corners_ws_path = r"c:\Users\TianyuZhao.LAPTOP-UNOKG67S\workspace\projects\extrinsic_cal_scan\edgeFOV\NAL_WS_analysis_08032022\{}_withWS\corners_and_correspondence.csv".format(wsIdx)
    #corners_gd_path = r"c:\Users\TianyuZhao.LAPTOP-UNOKG67S\workspace\projects\extrinsic_cal_scan\edgeFOV\NAL_WS_analysis_08032022\{}_post_noWS\corners_and_correspondence.csv".format(wsIdx) 
    npy_name = str(wsIdx)
    base_dir = r"c:\Users\TianyuZhao.LAPTOP-UNOKG67S\workspace\projects\extrinsic_cal_scan\edgeFOV\NAL_WS_analysis_08032022"

    corners_gd_list = []
    corners_ws_list = []
    for i in range(1, 4):
        for j in range(1, 10):
            wWSPath = os.path.join(base_dir, "{}-{}".format(i, j), "{}_withWS/corners_and_correspondence.csv".format(wsIdx))  #r"c:\Users\TianyuZhao.LAPTOP-UNOKG67S\Downloads\NAL_WS_testing_07272022\{}-{}\163007_withWS\corners_and_correspondence.csv".format(i, j)
            woWSPath = os.path.join(base_dir, "{}-{}".format(i, j), "{}_post_noWS/corners_and_correspondence.csv".format(wsIdx)) #r"c:\Users\TianyuZhao.LAPTOP-UNOKG67S\Downloads\NAL_WS_testing_07272022\{}-{}\163007_post_noWS\corners_and_correspondence.csv".format(i, j)
            print(i, j)
            corners_ws_list.append(pd.read_csv(wWSPath, header=None).to_numpy())
            corners_gd_list.append(pd.read_csv(woWSPath, header=None).to_numpy())
    corners_gd = np.concatenate(corners_gd_list, 0)
    corners_ws = np.concatenate(corners_ws_list, 0)
    # else:
    #     corners_gd = pd.read_csv(corners_gd_path, header=None).to_numpy()
    #     corners_ws = pd.read_csv(corners_ws_path, header=None).to_numpy()

    corners_gd_data.append(corners_gd)
    corners_ws_data.append(corners_ws)

    #estimate_distortion(corners_gd, corners_ws, npy_name)

# emsample all the corners into array
corners_gd = np.concatenate(corners_gd_data, 0)
corners_ws = np.concatenate(corners_ws_data, 0)

# feed into distortion training
estimate_distortion(corners_gd, corners_ws, "ensample")