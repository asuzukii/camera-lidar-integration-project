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

class Chessboard:
  """
  The Chessboard class takes in noisy scan of a chessboard and do corner feature 
    extraction.
  """
  def __init__(self, point_cloud_cluster, kernel, square_size, 
               num_square_vertical, num_square_horizontal, 
               width, height, index, padding, 
               resolution=0.005, display=False):

    self.point_cloud_cluster = point_cloud_cluster
    self.resolution = resolution
    self.kernel = kernel
    self.square_size = square_size
    self.num_square_v = num_square_vertical
    self.num_square_h = num_square_horizontal
    self.num_inner_v = self.num_square_v - 1
    self.num_inner_h = self.num_square_h - 1
    self.width = width
    self.height = height
    self.padding = padding
    self.index = index

    self.display = display

  def CornerExtraction(self, filter=True):
    """
    Process the data and perform the corner extraction.
    Input:
      filter [bool]: If true, filter the corners based on the grid given by config file
                     Otherwise, output all the potential corners.
    """
    valid_mask = self.outlier_filter(self.point_cloud_cluster, 0.075)
    self.point_cloud_cluster = self.point_cloud_cluster[valid_mask, :]

    self.image, self.pcs, self.offset, self.center_mean = self.projection_2D(self.point_cloud_cluster)
    self.image_normalized = self.interpolate_blank_pixels(self.image, 12)

    self.corners_image = self.feature_extraction(self.image_normalized)
    if filter:
      self.corners_image_filtered, self.correspond_target_corners = self.corner_filtering_corresponding(self.corners_image)                                             
      self.corners_3D = self.recover_3D(self.corners_image_filtered)
    else:
      self.corners_3D = self.recover_3D(self.corners_image)

  def outlier_filter(self, point_cloud, threshold=0.05):
    """
    Initial outlier filtering with ransac.
    Input:
      point_cloud [ndarray]: Input point cloud.
      threshold [float]: Outlier boundary, in meter.
    Output:
      valid_mask: a mask of valid points.
    """
    a, b, c, d = plane_detection_RANSAC(point_cloud)

    # Normalize the plane parameter
    w = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
    d /= np.sqrt(a**2 + b**2 + c**2)

    # Distance from each point to the plane
    d_to_plane = np.inner(w, point_cloud[:, :3]) + d
    
    # Discard points that are far away
    valid_points = np.abs(d_to_plane) <= threshold

    # Retrun a valid mask of data
    return valid_points

  def projection_2D(self, point_cloud_cluster):
    """
    Project point cloud from 3D to 2D
    Input:
      point_cloud_cluster [ndarray]: input point cloud 
    """

    w, d = plane_detection_L1(point_cloud_cluster[:, :3], 1)
    # Normalize the plane parameter
    d /= np.linalg.norm(w)
    w /= np.linalg.norm(w)

    # Distance from each point to the plane
    d_to_plane = np.inner(w, point_cloud_cluster[:, :3]) + d

    # Discard points that are far away
    valid_points = np.abs(d_to_plane) <= 0.05
    point_cloud_cluster = point_cloud_cluster[valid_points, :]
    d_to_plane = d_to_plane[valid_points]

    projection_2d = np.zeros_like(point_cloud_cluster)
    # Solve alpha*(ax + by + cz) + d = 0
    alpha = - d / np.inner(w, point_cloud_cluster[:, :3])
    #projection_2d[:, :3] = point_cloud_cluster[:, :3] - np.matmul(d_to_plane[:, None], w[None, :])
    projection_2d[:, :3] = alpha[:, None] * point_cloud_cluster[:, :3]
    projection_2d[:, 3] = point_cloud_cluster[:, 3]

    # Write to visualize
    csv_writer("./output/proj_{}.csv".format(self.index), projection_2d)

    # Project unit vector x axis on the plane
    # Use a rough estimation 
    x_1 = np.array([1, 0, 0])
    x_0 = np.array([0, 0, 0])
    origin_on_plane = (x_0 - w * (np.inner(w, x_0) + d))

    x_direction = (x_1 - w * (np.inner(w, x_1) + d)) - (x_0 - w * (np.inner(w, x_0) + d))
    x_unit = x_direction / np.linalg.norm(x_direction)

    # Compute unit vector z with cross multiplication. Vector w should be in incoming direction.
    z_unit = np.cross(np.sign(d)*w, x_unit)

    u = np.inner(projection_2d[:, :3], x_unit)
    v = np.inner(projection_2d[:, :3], z_unit)
    
    uv = np.stack([u, v], 1) 
    theta = bounding_box_regression(uv, np.array([self.width+2*self.padding, self.height+2*self.padding]))

    unit_hor = np.array([np.cos(theta), np.sin(theta)])
    unit_ver = np.array([-np.sin(theta), np.cos(theta)])

    coord_hor = np.inner(uv, unit_hor)
    coord_ver = np.inner(uv, unit_ver)
    coords_2d = np.stack([coord_hor, coord_ver], 1) 

    principle_components = np.zeros([2, 3])
    principle_components[0, :] = unit_hor[0] * x_unit + unit_hor[1] * z_unit
    principle_components[1, :] = unit_ver[0] * x_unit + unit_ver[1] * z_unit

    # Add the intensity
    data_2d = np.concatenate([coords_2d, projection_2d[:, 3:]], 1)

    # Move the origin to the bottom left corner
    offset = np.min(data_2d[:, :2], 0)
    data_2d[:, :2] -= offset

    image = points_to_image(data_2d, self.resolution)

    return image, principle_components, offset, origin_on_plane

  def recover_3D(self, corners_image):
    """
    Given the position of corners, in image coordinate. Recover it back to 3D 
      space.
    Input:
      corners [ndarray]:
      pcs [ndarray]: Principle components.
      offset:
      resolution
    """

    corners_2D = corners_image * self.resolution + np.array([self.resolution / 2, self.resolution / 2]) + self.offset

    # Matrix multiplication: (n, 1) * (1, 3) = (n, 3)
    corners_3d = corners_2D[:, 0, None] * self.pcs[None, 0, :] + corners_2D[:, 1, None] * self.pcs[None, 1, :]
    corners_3d += self.center_mean

    return corners_3d
    
  # Implemented in CPP in util.h
  def interpolate_blank_pixels(self, image, max_blank_size = 15):
    """
    Interpolates pixels with intensities of zero.
    Does this by setting intensity to the average of neighboring pixels
    Input:
      image [np.ndarray]: image
    Output:
      image [np.ndarray]: image
    """
    image = np.clip(image, 0, 1)
    OFFSETS = {(0, -1), (0, 1), (-1, 0), (1, 0)}
    blank_spots, num_spots = ndimage.label(image == 0)
    # For each cluster of blank pixels
    for i in range(1, num_spots + 1):
      blank_coords = np.stack(np.nonzero(blank_spots == i), axis=0)
      if len(blank_coords[0]) > max_blank_size:
        # This is the border of the chessboard, so don't interpolate
        continue
      
      num_vals = 0
      tot_vals = 0
      # Use set for O(1) search efficiency
      blank_coords_set = set(zip(blank_coords[0], blank_coords[1]))
      # Find the average intensity of all points that are adjacent to the blank pixels
      for point in blank_coords_set:
        for offset in OFFSETS:
          neighbor_point = (point[0] + offset[0], point[1] + offset[1])
          if neighbor_point not in blank_coords_set and \
              neighbor_point[0] >= 0 and neighbor_point[0] < image.shape[0] and \
              neighbor_point[1] >= 0 and neighbor_point[1] < image.shape[1]:
            num_vals = num_vals + 1
            tot_vals = tot_vals + image[neighbor_point[0]][neighbor_point[1]]

      # Replace blank pixels with this average
      image[blank_coords[0], blank_coords[1]] = tot_vals / num_vals

    return image


  def scale_gray_squares(self, image, gray_low_threshold=0.4, gray_high_threshold=0.8, square_size=20) -> np.ndarray:
    """
      Scales the intensity of gray squares to be closer to 'white'
      Input:
        image: np.ndarray
        gray_low_threshold: The minimum intensity to consider scaling
        gray_high_threshold: The max intensity to consider scaling
        square_size: The size of a square, in pixels.
      Output:
        image: np.ndarray: A scaled version of the input
    """

    # Locate coordinates where the pixel is gray
    grey_pts = np.stack(np.where((image > gray_low_threshold) & (image < gray_high_threshold)), 1)
    tree = KDTree(grey_pts)
    pts = []
    ss2 = square_size * square_size
    # If there are square_size^2 gray points near a point, then save that point to be scaled later
    for idx, neighbors in enumerate(tree.query_ball_point(grey_pts, square_size, workers=-1)):
      if len(neighbors) > ss2:
        pts.append(grey_pts[idx])
    # Scale the point using the cube root function
    if len(pts) > 0:
      pts = np.array(pts)
      # Try scale up, down, and no scaling, and see which does better.
      scale_up = np.copy(image)
      scale_down = np.copy(image)
      scale_up[pts[:, 0], pts[:, 1]] = np.cbrt(scale_up[pts[:, 0], pts[:, 1]])
      scale_down[pts[:, 0], pts[:, 1]] = np.power(scale_down[pts[:, 0], pts[:, 1]], 3)

      num_corners_scale_up = len(self.feature_extraction(scale_up, display=False))
      num_corners_scale_down = len(self.feature_extraction(scale_down, display=False))
      num_corners_original = len(self.feature_extraction(image, display=False))

      if num_corners_original > num_corners_scale_down and num_corners_original > num_corners_scale_up:
        return image
      if num_corners_scale_down > num_corners_scale_up:
        return scale_down
      elif num_corners_scale_up > num_corners_scale_down:
        return scale_up
    return image

  def image_noise_suppression(self, image, sample_size=3, square_size=20):
    """
    Removes noise from an image
    Input:
      Image: np.ndarray: The input image to remove noise from
      sample_size: int: How big of a radius to use for noise detection
      Square_size: int: The size of a square in the chessboard, in pixels
    output:
      Image: np.ndarray: Image after removing noise
    """
    min = ndimage.minimum_filter(image, size=sample_size)
    max = ndimage.maximum_filter(image, size=sample_size)
    variance = max - min > 0.5
    noisy_coords = np.stack(np.where(variance), 1)
    tree = KDTree(noisy_coords)
    ss2 = square_size * square_size
    pts = []

    # If there are square_size^2 noisy points near a point, then save that point
    for idx, neighbors in enumerate(tree.query_ball_point(noisy_coords, square_size, workers=-1)):
      if len(neighbors) > ss2:
        pts.append(noisy_coords[idx])

    # If noisy points are detected, try feature extraction on both min and max, and
    # use whichever gives the best results with feature extraction.
    if len(pts) > 0:
      pts = np.array(pts)
      min_img = np.copy(image)
      max_img = np.copy(image)
      min_img[pts[:, 0], pts[:, 1]] = min[pts[:, 0], pts[:, 1]]
      max_img[pts[:, 0], pts[:, 1]] = max[pts[:, 0], pts[:, 1]]

      corners_detected_min = self.feature_extraction(min_img, display=False)
      corners_detected_max = self.feature_extraction(max_img, display=False)
      corners_detected = self.feature_extraction(image, display=False)

      if len(corners_detected) > len(corners_detected_min) and len(corners_detected) > len(corners_detected_max):
        return image
      elif len(corners_detected_max) < len(corners_detected_min):
        return min_img
      elif len(corners_detected_max) > len(corners_detected_min):
        return max_img
    
    return image


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
    min = conv_output
    threshold = 0.70 * np.amax(min)
    min = min > threshold
    labeled, num_objects = ndimage.label(min)

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
    # CPP- See cepton_ext_cal::fuse_nearby_points
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

  def corner_filtering_corresponding(self, corners_image):
    """
    Use ICP to match the known patter with the detected corner to:
      1. Filter the outliers and find the correspondence
      2. Sort the corners in certain order
    """
    pixel_per_square = self.square_size / self.resolution
    points_pattern = []
    for j in reversed(range(self.num_inner_v)):
      for i in range(self.num_inner_h):
        points_pattern.append(np.array([i*pixel_per_square, j*pixel_per_square]))

    point_ref = np.stack(points_pattern, 0)

    valid_ref, valid_detect = ICP_correspondence(point_ref, corners_image, pixel_per_square)

    return corners_image[valid_detect, :], valid_ref
