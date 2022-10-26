"""
Calibrate the intrinsic of the camera.
Evaluate the calibration via re-projection error
Write the undistorted image to file
https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
"""

# TODO: make good error handling

import numpy as np
import cv2
import glob
import io
import matplotlib.pyplot as plt
import os
import argparse


def generate_checkerboard_coords(checkerboard_size: tuple):
    """
    Get the checkerboard coordinates in 3D space,
    with the origin being bottom-left corner.
    Input: checkerboardSize: (num_row, num_col)
           num_row: rows in the checkerboard
           num_col: column in the checkerboard
    Output: numpy.array: (num_row * num_col, 3). with Z coordinate always 0
    """
    num_row, num_col = checkerboard_size
    checkerboard_coords = np.zeros((num_row * num_col, 3), np.float32)
    checkerboard_coords[:, :2] = np.mgrid[0:num_row, 0:num_col].T.reshape(-1, 2)

    return checkerboard_coords

# TODO: function description
def camera_intrinsic_calibration(image_size, checkerboard_size, image_paths):
    valid_checkerboard_coords = []
    image_points_list = []
    img_criteria = (
          cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image_path in image_paths:
      print("Processing image: {}".format(image_path))
      image = cv2.imread(image_path)
      grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      checkerboard_coords = generate_checkerboard_coords(checkerboard_size)

      # Find the checkerboard corners
      ret, checkerbaord_corners = cv2.findChessboardCorners(grayscale_image, checkerboard_size, None)

      if ret:
          valid_checkerboard_coords.append(checkerboard_coords)
          corners_improved = cv2.cornerSubPix(
              grayscale_image, checkerbaord_corners, (11, 11), (-1, -1), img_criteria)

          image_points_list.append(corners_improved)

          # Draw corners for Debugging
          cv2.drawChessboardCorners(
              image, checkerboard_size, corners_improved, ret)
          # TODO: make the window popup a little better when images are too big
          cv2.imshow(image, image)
          cv2.waitKey(500)
          cv2.destroyAllWindows()
      else:
          raise ValueError("not detected")
    # cv2.destroyAllWindows()

  # Calibration
    return cv2.calibrateCamera(
        valid_checkerboard_coords, image_points_list, image_size, None, None)

# TODO: function description
def write_results_to_file(save_dir, image_paths, calibration_results):
    ret, mtx, dist, rvecs, tvecs = calibration_results
    with io.open(os.path.join(save_dir, 'camera_calibration_result.txt'), 'w') as f:
        f.write("RMS error:\n")
        np.savetxt(f, np.array(ret).reshape((1,)))

        f.write("Camera Matrix:\n")
        np.savetxt(f, mtx)

        f.write("Distortion Coefficient:\n")
        np.savetxt(f, dist)

    np.savez(os.path.join(save_dir, 'camera_calibration_result.npz'), RMS=np.array(
        ret).reshape((1,)), camera_matrix=mtx, distortion_coeff=dist)

    # Undistortion
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, image_size, 0, image_size)

    undistorted_dir = save_dir
    if not os.path.exists(undistorted_dir):
        os.mkdir(undistorted_dir)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_undistorted = cv2.undistort(image, mtx, dist, None, newCameraMatrix)
        x, y, w, h = roi
        image_undistorted = image_undistorted[y:y+h, x:x+w]
        cv2.imwrite(
            os.path.join(
                undistorted_dir, 'undistorted_{}'.format(
                    os.path.basename(image_path))),
            image_undistorted)

def main():
  # TODO: make descriptions for the arguments
  # TODO: fix the required/typing of the arguments
    parser = argparse.ArgumentParser(
        description='Camera Intrinsic Calibration')
    parser.add_argument('--image-dir', '-d', type=str, required=True)
    parser.add_argument('--save-dir', '-s', type=str)
    parser.add_argument('--image-width', '-w', type=str, required=True)
    parser.add_argument('--image-height', '-h', type=str, required=True)
    parser.add_argument('--show-corners', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.image_dir):
        raise argparse.ArgumentError(
            "Specified folder: {} doesn't exist.".format(args.image_dir))

    print("Starting Intrinsic Calibration Process...")

    image_size = args.image_width, args.image_height
    checkerboard_size = args.checkerboard_width, args.checkerboard_height
    images = glob.glob(os.path.join(args.image_dir, '*.png'))

    calibration_results = camera_intrinsic_calibration(image_size, checkerboard_size, images)

    if args.save_dir:
        write_results_to_file(args.save_dir, args.image_dir, calibration_results)

if __name__ == '__main__':
    main()
