###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import os
import argparse

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import pathlib
from os.path import abspath, splitext

from cepton_extrinsic_calibration.data_io import csv_writer, pcap_reader
from cepton_extrinsic_calibration.bigwall import BigWall
from cepton_extrinsic_calibration.preprocess import preprocess

import cepton_extrinsic_calibration
print("##### Using Cepton Extrinsic Calibration Version: {} #####".format(cepton_extrinsic_calibration.__version__))

if __name__ == "__main__":
  # Read command line arguments  
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--file', type=str, help='input csv/pcap file', required=True)
  parser.add_argument('-o', '--output_directory', type=str, help='output base directory', 
                      default='./output/', required=False)
  parser.add_argument('--display', action='store_true', help='display plot')
  parser.add_argument('-c', '--config', type=str, help="path to config file", required=False)
  args = parser.parse_args()

  file_path = args.file
  stem_path, extension = splitext(file_path)
  stem_name = pathlib.Path(file_path).stem
  if args.config is None:
    json_path = stem_path + ".json"
  else:
    json_path = args.config

  if not os.path.exists(file_path):
    raise FileNotFoundError("Input point cloud data not found")
  if not os.path.exists(json_path):
    raise FileNotFoundError("Input meta data not found")

  if not os.path.exists(args.output_directory):
    os.mkdir(args.output_directory)

  save_dir = os.path.join(args.output_directory, stem_name)
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  if extension == ".csv":
    try:
      csv_file = pd.read_csv(file_path, usecols=('x','y','z','reflectivity','valid',
                                          'saturated','segment_id'), 
                      dtype={'x': np.float64, 'y': np.float64, 'z': np.float64, 
                              'reflectivity': np.float64, 'valid': np.bool8, 
                              'saturated':np.bool8, 'segment_id': np.int8})
    except:
      print("No header. Assume x, y, z, reflectivity, valid, saturated and segment_id.")
      csv_file = pd.read_csv(file_path)
    data = csv_file.to_numpy()
  elif extension == ".pcap":
    print("Reading data from {}".format(abspath(file_path)))
    data = pcap_reader(file_path)
  else:
     raise ValueError("Unknown extension: {}".format(extension))

  kernel = np.load("./cepton_extrinsic_calibration/kernels/25kernel_smoother.npy")

  # Meta data should at the same directory and has the same name (.json extension)
  print("Reading meta data from {}".format(abspath(json_path)))
  with open(json_path) as f:
    metadata = json.load(f)
  chessboard_spec = metadata["chessboard"]
  num_chessboard = len(chessboard_spec)

  #compute_each_corner(chessboard_spec)

  print("Preprocessing data...")
  point_cloud = preprocess(data, *metadata["boundary"])
  print("Number of points after preprocessing: {}".format(len(point_cloud)))

  print("Clustering data points to differentiate chessboards...")
  #labels, centers, valid_clusters = clustering_connectivity(point_cloud, num_chessboard)

  # For each cluster (in ascending order by x-axis value)
  cluster_order = [0]#np.argsort(centers[:, 0])

  corner_scan = []; corner_gd = []; corner_lengths = []; target_layout = []
  corners_and_correspondence = []
  valid_corners = []; cover_area = []
  for i, cluster_idx in enumerate(cluster_order):
    print("Target: {}".format(i))
    #point_cloud_cluster = point_cloud[labels==valid_clusters[cluster_idx], :]
    point_cloud_cluster = point_cloud#[labels==valid_clusters[cluster_idx], :]
    
    # meta data
    square_size = chessboard_spec[i]["square_size"]
    num_horizontal = chessboard_spec[i]["num_horizontal"]
    num_vertical = chessboard_spec[i]["num_vertical"]
    padding = chessboard_spec[i]["padding"]

    width = chessboard_spec[i]["x_tick"][-1]
    height = chessboard_spec[i]["z_tick"][-1]
    cover_area.append(width*height)

    target_layout.append([num_vertical-1, num_horizontal-1])

    # from cepton_extrinsic_calibration.modules.c_preprocess import CornerExtractionCpp
    # image = CornerExtractionCpp(point_cloud_cluster, kernel, i, square_size, num_vertical, num_horizontal, height, width)

    chessboard = BigWall(point_cloud_cluster, kernel, square_size, 
                            num_vertical, num_horizontal, width, height, 
                            i, padding, 0.005, display=args.display)
    chessboard.CornerExtraction()

    # Save results of computations to file
    corners_and_correspondence_index = np.concatenate([chessboard.corners_3D, chessboard.correspond_target_corners[:, None]], 1)
  
    all_corners_index = np.arange((len(chessboard_spec[i]["x_tick"]) - 2) * (len(chessboard_spec[i]["z_tick"]) - 2))

    target_number = np.ones([len(corners_and_correspondence_index), 1]) * i
    corners_and_correspondence_index = np.concatenate([corners_and_correspondence_index, target_number], 1)
    corners_and_correspondence.append(corners_and_correspondence_index)
    detected_index = corners_and_correspondence_index[:, 3]

    valid_corners_index = all_corners_index[detected_index.astype(np.int32)]
    valid_corners.append(valid_corners_index)

    corner_scan.append(corners_and_correspondence_index[:, :3])
    corner_lengths.append(len(valid_corners_index))

    fig = plt.figure(figsize=(30,10))
    plt.imshow(chessboard.image_normalized.T, origin="lower")
    plt.title("{}_{}".format(stem_name, i))
    plt.scatter(chessboard.corners_image[:, 0], chessboard.corners_image[:, 1], c='w')
    plt.savefig(os.path.join(save_dir, "image{}.png".format(i)))
  
  corner_scan = np.concatenate(corner_scan, 0)
  valid_corners = np.concatenate(valid_corners, 0)
  corners_and_correspondence = np.concatenate(corners_and_correspondence, 0)

  csv_writer(os.path.join(save_dir, "corners_and_correspondence.csv"), corners_and_correspondence)

  if args.display:
    plt.show()