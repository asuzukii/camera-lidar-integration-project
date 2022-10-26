###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import pathlib
from os.path import abspath, splitext

from cepton_extrinsic_calibration.data_io import csv_writer, pcap_reader
from cepton_extrinsic_calibration.chessboard import Chessboard
from cepton_extrinsic_calibration.data_io import csv_reader, export_parameter_to_json, read_json
from cepton_extrinsic_calibration.clustering import clustering_connectivity
from cepton_extrinsic_calibration.preprocess import preprocess, compute_each_corner
from cepton_extrinsic_calibration.registration import registration, evaluate_parameters, error_analysis
from cepton_extrinsic_calibration.models.distortion import nominal_parameters
from cepton_extrinsic_calibration.coordinate_change import coord_rotation_VICON_to_sensor

import cepton_extrinsic_calibration
print("##### Using Cepton Extrinsic Calibration Version: {} #####".format(cepton_extrinsic_calibration.__version__))

plt.rcParams.update({'font.size': 14})

if __name__ == "__main__":
  # Read command line arguments  
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--file', type=str, help='Input csv/pcap file', required=True)
  parser.add_argument('-m', '--mode', type=str, help='Run the code in following mode: \
      train, validate, param-in(parameter already loaded in the sensor)', required=True)
  parser.add_argument('--load_param', type=str, help='The parameter JSON file to load in \
      validation and param-in mode', required=False, default="")
  parser.add_argument('--display', action="store_true", help='Display plot when this flas \
      is set', required=False)
  args = parser.parse_args()

  if not args.mode in ["train", "validate", "param-in"]:
    raise ValueError("Unknown mode")

  # data IO
  path = args.file
  stem_path, extension = splitext(path)
  stem_name = pathlib.Path(path).stem

  if extension == ".csv":
    print("Reading data from {}".format(abspath(path)))
    data = csv_reader(path)
  elif extension == ".pcap":
    print("Reading data from {}".format(abspath(path)))
    data = pcap_reader(path)
    parent_dir = pathlib.Path(path).parents[0]
    save_csv_path = os.path.join(parent_dir, stem_name+".csv")
    if not os.path.exists(save_csv_path):
      csv_writer(save_csv_path, data, 
        ["x", "y", "z", "reflectivity", "valid", "saturated", "segment_id"])
  else:
     raise ValueError("Unknown extension: {}".format(extension))
  
  kernel = np.load("./cepton_extrinsic_calibration/kernels/25kernel_smoother.npy")

  if not os.path.exists("./output"):
    os.mkdir("./output")

  save_dir = os.path.join("./output/", stem_name)
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  # Ground truth data
  # Meta data should at the same directory and has the same name (.json extension)
  json_path = stem_path + ".json"
  print("Reading meta data from {}".format(abspath(json_path)))
  with open(json_path) as f:
    metadata = json.load(f)
  chessboard_spec = metadata["chessboard"]
  num_chessboard = len(chessboard_spec)

  compute_each_corner(chessboard_spec)

  # Preprocess
  print("Preprocessing data...")
  point_cloud = preprocess(data, *metadata["boundary"])

  print("Number of points after preprocessing: {}".format(len(point_cloud)))

  print("Clustering data points to differentiate chessboards...")
  labels, centers, valid_clusters = clustering_connectivity(point_cloud, num_chessboard)

  # For each cluster (in ascending order by x-axis value)
  cluster_order = np.argsort(centers[:, 0])

  # Bring scan back to sensor origin
  if args.mode in ["param-in"]:
    parameter_dict = read_json(args.load_param)
    transformation_matrix = np.array(parameter_dict["transformation"])
    transformation_matrix = np.reshape(transformation_matrix, [4,4])
    inv_transformation_matrix = np.linalg.inv(transformation_matrix)

  # Main loop to iterate each of the chessboard for corner feature extraction
  corner_scan = []; corner_gd = []; corner_lengths = []; target_layout = []
  corners_and_correspondence = []
  valid_corners = []; cover_area = []
  for i, cluster_idx in enumerate(cluster_order):
    print("Target: {}".format(i))
    point_cloud_cluster = point_cloud[labels==valid_clusters[cluster_idx], :]
  
    if args.mode in ["param-in"]:
      point_cloud_cluster[:, :3] = (inv_transformation_matrix[:3, :3] @ point_cloud_cluster[:, :3].T + \
                                    inv_transformation_matrix[:3, 3, None]).T

    # meta data
    square_size = chessboard_spec[i]["square_size"]
    num_horizontal = chessboard_spec[i]["num_horizontal"]
    num_vertical = chessboard_spec[i]["num_vertical"]
    padding = chessboard_spec[i]["padding"]

    width = chessboard_spec[i]["x_tick"][-1]
    height = chessboard_spec[i]["z_tick"][-1]
    cover_area.append(width*height)

    target_layout.append([num_vertical-1, num_horizontal-1])

    chessboard = Chessboard(point_cloud_cluster, kernel, square_size, 
                            num_vertical, num_horizontal, width, height, 
                            i, padding, 0.005, display=True)
    chessboard.CornerExtraction()
    
    # Save results of computations to file
    corners_and_correspondence_index = np.concatenate([chessboard.corners_3D, chessboard.correspond_target_corners[:, None]], 1)
    csv_writer(os.path.join(save_dir, "cluster{}.csv".format(i)), point_cloud_cluster)

    all_corner_gd = np.array(chessboard_spec[i]["corners"])
  
    all_corners_index = np.arange(len(chessboard_spec[i]["corners"]))

    target_number = np.ones([len(corners_and_correspondence_index), 1]) * i
    corners_and_correspondence_index = np.concatenate([corners_and_correspondence_index, target_number], 1)
    corners_and_correspondence.append(corners_and_correspondence_index)
    detected_index = corners_and_correspondence_index[:, 3]

    valid_corners_index = all_corners_index[detected_index.astype(np.int32)]
    valid_corners.append(valid_corners_index)

    corner_scan.append(corners_and_correspondence_index[:, :3])
    corner_gd.append(all_corner_gd[valid_corners_index, :3])
    corner_lengths.append(len(valid_corners_index))
  
  corner_scan = np.concatenate(corner_scan, 0)
  corner_gd = np.concatenate(corner_gd, 0)
  valid_corners = np.concatenate(valid_corners, 0)
  corners_and_correspondence = np.concatenate(corners_and_correspondence, 0)

  # Save feature extraction results
  csv_writer(os.path.join(save_dir, "corners_and_correspondence.csv"), corners_and_correspondence)
  csv_writer(os.path.join(save_dir, "corner_gd.csv"), corner_gd)

  if args.mode == "train":
    # In training mode, do registration and export parameters
    parameters = registration(corner_scan, corner_gd, corner_lengths, display=True)
    print(parameters)
    export_parameter_to_json(parameters, save_dir, "parameters")

  elif args.mode == "validate":
    # In validation mode, load an existing parameters
    parameters = read_json(args.load_param)
    parameters["transformation"] = np.reshape(np.array(parameters["transformation"]), [4, 4])
    parameters["distortion"] = np.array(parameters["distortion"])
    print(parameters)

  elif args.mode == "param-in":
    # In param-in mode, load a set of parameters that do identity transformation
    corner_gd = coord_rotation_VICON_to_sensor(corner_gd)
    corner_gd = (inv_transformation_matrix[:3, :3] @ corner_gd.T + inv_transformation_matrix[:3, 3, None]).T
             
    parameters = nominal_parameters()

  # Evaluation
  error_in, error_out, ref_angular = evaluate_parameters(corner_scan, corner_gd, corner_lengths, parameters, save_dir)
  error_analysis(error_in, error_out, ref_angular, valid_corners, corner_lengths, target_layout, save_dir, args.mode)

  # Display results
  if args.display:
    plt.show()