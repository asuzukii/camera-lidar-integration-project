# Intrinsic Camera Calibration

## Get Started
IntrinsicCameraCalibration.py will allow you to get rid of the lens distortion
from your camera, undistorting the images in the process. To get started, from this directory (`canera-lidar-integration-project/camera_intrinsic_calibration`), try:

`python IntrinsicCameraCalibration.py -d <absolute/path/for/checkerboard/images> -s <absolute/path/for/undistorted/images/to/go> -iw <image_width> -ih <image_height> -cw <checkerboard_width-1> -ch <checkerboard_height-1>`

This will save undistorted images to the specified directory, as well as outputting camera matrix and distortion coefficient information into a file.
A npz file is also created with all of these parameters.

If you would want to see it with a real example, use the images included in this subrepo with:
`python IntrinsicCameraCalibration.py -d <absolute/path/for/checkerboard/images> -s <absolute/path/for/undistorted/images/to/go> -iw 2048 -ih 1536 -cw 9 -ch 7`

## Tips for Getting Good Calibration Parameters: 
Here are a couple of tips to get your camera system to be properly working in the first try:

- The `ret` from the `calibrateCamera()` function is the mean reprojection error, i.e. the average distance between the original corner points and the undistorted corner points. Anything below a 0.30 reprojection error is almost necessary but not sufficient.
- To keep the orientations of the `findChessboardCorners(),` try not to tilt the chessboard pattern more than 45 degrees from the camera. This will allow for consistent orientation of how the opencv module sees the chessboard, making the camera matrix more accurate.
- When using cameras with different dimensions, calibrate them with same dimensions, each cropping the biggest centered rectangle possible (e.g. if you have a 1280x1024 and 2048x1690 camera, you can take a 1280x1024 centered crop of the 2048x1690 camera (make sure it’s centered!))
- Make sure the checkerboard used for the calibration is completely flat, and the print is clear. If unsure, using a tablet is a good alternative to printed out boards (even though then there’s some glare that you’d have to be careful of)
