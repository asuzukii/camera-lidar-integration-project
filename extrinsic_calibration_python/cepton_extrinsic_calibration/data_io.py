###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
import os
from PIL import Image
import pandas as pd
import cepton_sdk2 as sdk
import time
import json
import transforms3d

from .models.distortion import parameter_unpack, NUM_DISTORTION_PARAM
from .coordinate_change import coord_rotation_VICON_to_sensor

def csv_reader_old(file: str, header=None):
  """
  Read from a csv file
  Input: 
    file [string]: path to the csv file 
  Return: a list of points. Should be parsed by preprocess afterwards.
  """
  csv_file = pd.read_csv(file, header=None)
  return csv_file.to_numpy()

def csv_reader(file: str):
  """
  Read from a csv file
  Input: 
    file [string]: path to the csv file 
  Return: a ndarray of points. Should be parsed by preprocess afterwards.
  """
  csv_file = pd.read_csv(file, usecols=('x','y','z','reflectivity','valid',
                                        'saturated','segment_id'), 
                    dtype={'x': np.float64, 'y': np.float64, 'z': np.float64, 
                            'reflectivity': np.float64, 'valid': np.bool8, 
                            'saturated':np.bool8, 'segment_id': np.int8})
  return csv_file.to_numpy()

def csv_writer(file: str, data: np.ndarray, labels=None):
  """
  Write the point cloud data to csv file
  Input: 
    file [string]: path to save the csv file
    data [ndarray]: point cloud data, n * 4
    labels [list(str)]: List of labels for the data
  """
  if not os.path.exists(os.path.dirname(os.path.abspath(file))):
    os.makedirs(os.path.dirname(os.path.abspath(file)))
  pd.DataFrame(data).to_csv(file, index=None, header=False if not labels else labels)

def pcap_reader(file: str):
  """
  Read pcap file with cepton sdk and export point cloud in following order:
  x, y, z, reflectivity, valid, saturated, segment_id
  """
  # Variables
  capture_path = file # put path to pcap here
  # Initialize
  sdk.Initialize()
  # LoadPcap
  # speed=100 means 1x speed. Speed=0 means as fast as possible.
  sdk.LoadPcap(capture_path, speed=0)
  # Enable FIFO feature
  # Frame aggregation mode set to 0(natrual). Allocate 400 frame buffers in the frame FIFO
  sdk.EnableFrameFifo(frameMode=0, nFrames=400)

  frames = []
  frame_count = 0
  # Loop until pcap replay is finished
  while not sdk.ReplayIsFinished() or not sdk.FrameFifoEmpty():
      frame = sdk.FrameFifoGetFrame(timeout=2000) # 2000 ms

      if not frame is None:
          frames.append(frame)
          frame_count += 1
          sdk.FrameFifoRelease()     

  # Disable FIFO feature
  sdk.DisableFrameFifo()
  # Deinitialize
  sdk.Deinitialize()

  positions = [frame.positions for frame in frames]
  reflectivities = [frame.reflectivities for frame in frames]
  channel_id = [frame.channel_ids for frame in frames]
  invalid = [frame.invalid for frame in frames]
  saturated = [frame.saturated for frame in frames]
  boundary = [frame.frame_boundary for frame in frames]

  positions_array = np.concatenate(positions, 0)
  reflectivities_array = np.concatenate(reflectivities, 0)
  channel_id_array = np.concatenate(channel_id, 0)
  invalid_array = np.concatenate(invalid, 0)
  valid_array = invalid_array == 0
  saturated_array = np.concatenate(saturated, 0)
  boundary_array = np.concatenate(boundary, 0)

  mask = ~boundary_array
  point_cloud = np.concatenate([positions_array[mask], 
                                reflectivities_array[mask, None], 
                                valid_array[mask, None], 
                                saturated_array[mask, None], 
                                channel_id_array[mask, None].astype(np.int8)], 1)
  return point_cloud

def save_image_to_file(image: np.ndarray, filename: str, directory="./"):
  """
  Write an image (as a numpy array) to a file
  Input: 
    file [string]: path to save the csv file
    data [ndarray]: point cloud data, n * 4
  """
  Image.fromarray(image).convert('RGB').save(os.path.join(directory,filename))

def export_parameter_to_json(parameters, dir, name):
  """
  Save a parameter vector into json file.
  """

  print("exporting parameters to {}".format(dir))

  parameters_dict = {}
  transformation_matrix_list = parameters["transformation"].tolist()
  parameters_dict["transformation"] = transformation_matrix_list[0] + \
                                      transformation_matrix_list[1] + \
                                      transformation_matrix_list[2] + \
                                      transformation_matrix_list[3]
  parameters_dict["distortion"] = parameters["distortion"].tolist()

  with open(os.path.join(dir, name+"_for_VICON.json"), 'w') as f:
    json.dump(parameters_dict, f, indent=4)

  transformation_matrix_for_sensor = parameters["transformation"].copy()
  transformation_matrix_for_sensor[:3, :] = coord_rotation_VICON_to_sensor(transformation_matrix_for_sensor[:3, :].T).T
  transformation_matrix_list = transformation_matrix_for_sensor.tolist()

  parameters_dict["transformation"] = transformation_matrix_list[0] + \
                                      transformation_matrix_list[1] + \
                                      transformation_matrix_list[2] + \
                                      transformation_matrix_list[3]

  with open(os.path.join(dir, name+"_for_sensor.json"), 'w') as f:
    json.dump(parameters_dict, f, indent=4)

def read_json(file):
  with open(file) as f:
    data = json.load(f)
  return data