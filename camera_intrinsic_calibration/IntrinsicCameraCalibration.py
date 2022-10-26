"""
Calibrate the intrinsic of the camera.
Evaluate the calibration via re-projection error
Write the undistorted image to file
https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
"""

import numpy as np
import cv2
import glob
import io
import matplotlib.pyplot as plt
import os
import argparse


def generate_checkerboard_coords(checkerboard_size: tuple[int, int]):
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
def camera_intrinsic_calibration(image_size: tuple[int, int], 
                                checkerboard_size: tuple[int, int],
                                image_paths: list[str]):
    valid_checkerboard_coords = []
    image_points_list = []
    img_criteria = (
          cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # find corners for each image
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
      else:
          raise ValueError("Corners not detected")

    # Calibration
    calibration_result = cv2.calibrateCamera(
        valid_checkerboard_coords, image_points_list, image_size, None, None)
    print("===================================================")
    print("RMS (< 0.3 is good normally):", calibration_result[0])
    print("Camera Matrix:\n", calibration_result[1])
    print("Distortion Coefficients (k1 k2 p1 p2 k3):", calibration_result[2])
    print("===================================================")
    # verbose so not printing these
    # print("R Vec:", calibration_result[3])
    # print("T Vec:", calibration_result[4])

    return calibration_result

# TODO: function description
def write_results_to_file(save_dir: str, calibration_results: tuple):
    print("Results are being written to:", save_dir)
    ret, mtx, dist, rvecs, tvecs = calibration_results

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with io.open(os.path.join(save_dir, 'camera_calibration_result.txt'), 'w') as f:
        f.write("RMS error:\n")
        np.savetxt(f, np.array(ret).reshape((1,)))

        f.write("Camera Matrix:\n")
        np.savetxt(f, mtx)

        f.write("Distortion Coefficient:\n")
        np.savetxt(f, dist)

    np.savez(os.path.join(save_dir, 'camera_calibration_result.npz'), RMS=np.array(
        ret).reshape((1,)), camera_matrix=mtx, distortion_coeff=dist)

def undistort_and_save_images(camera_matrix,
                              distortion_coefficients,
                              image_paths: list[str],
                              image_size,
                              save_dir: str):
    newCameraMatrix, region_of_interest = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coefficients, image_size, 0, image_size)

    print("Undistorted Images being written to:", save_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_undistorted = cv2.undistort(image,
                                          camera_matrix,
                                          distortion_coefficients,
                                          None,
                                          newCameraMatrix)
        x, y, w, h = region_of_interest
        image_undistorted = image_undistorted[y:y+h, x:x+w]
        cv2.imwrite(
            os.path.join(save_dir,
                        'undistorted_{}'.format(os.path.basename(image_path))),
                        image_undistorted)

def main():
    parser = argparse.ArgumentParser(
        description='Camera Intrinsic Calibration')
    parser.add_argument('--image-dir', '-d', type=str, required=True,
                        help='directory of where the uncalibrated checkerboard pictures are')
    parser.add_argument('--save-dir', '-s', type=str,
                        help='directory of where calibrated checkerboard pictures\
                        and parameters output file should go to')
    parser.add_argument('--image-width', '-iw', type=int, required=True,
                        help='image width in pixels')
    parser.add_argument('--image-height', '-ih', type=int, required=True,
                        help='image width in pixels')
    parser.add_argument('--checkerboard-width', '-cw', type=int, required=True,
                        help='checkerboard width in squares')
    parser.add_argument('--checkerboard-height', '-ch', type=int, required=True,
                        help='checkerboard height in squares')

    args = parser.parse_args()

    # TODO: make more robust error handling here
    if not os.path.exists(args.image_dir):
        raise argparse.ArgumentTypeError(f"Specified folder: {args.image_dir} doesn't exist.")

    print("Starting Intrinsic Calibration Process...")

    image_size = args.image_width, args.image_height
    # calibration algorithm actually only cares about the internal corners of the board
    checkerboard_size = args.checkerboard_width-1, args.checkerboard_height-1
    image_paths = glob.glob(os.path.join(args.image_dir, '*.png'))

    calibration_results = camera_intrinsic_calibration(image_size,
                                                      checkerboard_size,
                                                      image_paths)

    if args.save_dir:
        write_results_to_file(args.save_dir,
                              calibration_results)
        undistort_and_save_images(calibration_results[1],
                        calibration_results[2],
                        image_paths,
                        image_size,
                        args.save_dir)

if __name__ == '__main__':
    main()
