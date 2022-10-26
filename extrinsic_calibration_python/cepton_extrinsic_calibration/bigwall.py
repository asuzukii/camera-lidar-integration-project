###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
import cv2
from scipy import ndimage
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from .projection import plane_detection_L1, plane_detection_RANSAC, points_to_image
from .icp import ICP_correspondence
from .linear_regression_L1 import bounding_box_regression
from .fit_gaussian import fit_gaussian_2D
from .data_io import csv_writer
from .chessboard import Chessboard

class BigWall(Chessboard):
  """
  The Chessboard class takes in noisy scan of a chessboard and do corner feature 
    extraction.
  """
  def __init__(self, point_cloud_cluster, kernel, square_size, 
               num_square_vertical, num_square_horizontal, 
               width, height, index, padding, 
               resolution=0.005, display=False):
    Chessboard.__init__(self, point_cloud_cluster, kernel, square_size, 
                num_square_vertical, num_square_horizontal, 
                width, height, index, padding, 
                resolution, display)


  def feature_extraction(self, image, display=None):
    """
    Corner feature extraction
    Input:
      image [ndarray]: 2D image. Array as size (m, n).
      display [bool]: Whether to display or not. True or False will 
        override self.default. If display==None, then use self.display.
    Return:
      corners [ndarray]: Array of corner coordinates
    """

    # OpenCV in CPP is set up and ready to use.
    conv_output = cv2.filter2D(image, -1, self.kernel)

    conv_output = np.abs(conv_output)

    nSlices = 8
    sliceLength = int(np.ceil(conv_output.shape[0] / nSlices))
    sliceMasks = []
    for i in range(nSlices):
      threshold = 0.60 * np.amax(conv_output[i*sliceLength:(i+1)*sliceLength, :])
      sliceMasks.append(conv_output[i*sliceLength:(i+1)*sliceLength, :] > threshold)
    
    peakMask = np.concatenate(sliceMasks, 0)

    labeled, num_objects = ndimage.label(peakMask)

    peaks = []; scales = []
    for obj_idx in range(1, num_objects+1):
      coords = np.stack((labeled == obj_idx).nonzero(), 1)
      peak, scale = fit_gaussian_2D(coords, conv_output[coords[:, 0], coords[:, 1]])

      peaks.append(peak)
      scales.append(scale)
    peaks = np.array(peaks)
    scales = np.array(scales)

    # Find peak index
    corners_array = peaks

    # Fuse points that are nearby using DBSCAN Clustering
    cluster_radius = 15
    db_labels = DBSCAN(eps=cluster_radius, min_samples=2).fit_predict(corners_array)
    fused_points = corners_array[db_labels==-1].tolist() # Peaks without nearby peaks
    for i in range(0, np.amax(db_labels) + 1):
      points_to_merge = corners_array[db_labels==i]
      maximum_peak_idx = np.argmax(scales[db_labels==i])
      fused_points.append(points_to_merge[maximum_peak_idx, :].tolist())
    
    corners_array = np.array(fused_points)
    
    if (display is not None and display) or (display is None and self.display):
      _, ax1 = plt.subplots()
      ax1.imshow(conv_output.T, origin="lower")
      ax1.set_title("Conv. Output, Max: {}".format(np.max(conv_output)))

      _, ax2 = plt.subplots()
      ax2.imshow(labeled.T, origin="lower")
      ax2.set_title("Thresholded Conv. Output")
      
      _, ax3 = plt.subplots()
      ax3.imshow(image.T, origin="lower")
      ax3.scatter(corners_array[:, 0], corners_array[:, 1], c='w')
      ax3.set_title("Plotted Corners")

      fig = plt.figure()
      plt.imshow(image.T, origin="lower")
      plt.scatter(corners_array[:, 0], corners_array[:, 1], c='w')
      plt.savefig("./output/image{}.png".format(self.index))
      plt.close(fig)

    return corners_array
