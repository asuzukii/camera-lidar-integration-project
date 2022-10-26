###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import os
import argparse
import numpy as np
import pandas as pd
import pathlib
from os.path import abspath, splitext

from cepton_extrinsic_calibration.data_io import csv_writer, pcap_reader
from cepton_extrinsic_calibration.data_io import read_json
from cepton_extrinsic_calibration.models.distortion import apply_bezier_distortion
from cepton_extrinsic_calibration.models.bezier import BezierModel

import cepton_extrinsic_calibration
print("##### Using Cepton Extrinsic Calibration Version: {} #####".format(cepton_extrinsic_calibration.__version__))

def offline_transformation(input, param):

  input = input.astype(np.float64)

  transform_matrix = param["transformation"]
  distortion_parameters = param["distortion"]

  model = BezierModel()
  model.load_parameter(distortion_parameters)

  R = transform_matrix[:3, :3]
  translation = transform_matrix[:3, 3]

  source_undistorted = apply_bezier_distortion(input, model)

  source_undistorted_transformed = (R@source_undistorted.T + translation[:, None]).T

  return source_undistorted_transformed

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--file', type=str, required=True)
  parser.add_argument('-c', '--config', type=str, required=True)
  parser.add_argument('-o', '--output', type=str, required=True)
  args = parser.parse_args()

  parameter_dict = read_json(args.config)
  parameter_dict["transformation"] = np.reshape(parameter_dict["transformation"], (4, 4))

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

  input_point_cloud = data
  print(len(input_point_cloud))

  output_point_cloud = input_point_cloud.copy()
  print("Transforming...")
  output_point_cloud[:, :3] = offline_transformation(input_point_cloud[:, :3], parameter_dict)

  data[:, :3] = output_point_cloud[:, :3]

  print("Writing to file")
  df = pd.DataFrame(data)
  df.to_csv(args.output, index=False, header=["x", "y", "z", "reflectivity", "saturated", "valid", "segment_id"])
