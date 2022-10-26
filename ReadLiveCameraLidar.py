import os
import cv2
import sys
import argparse

from PcapCollection import externalWiresharkCollection

def main():
  parser = argparse.ArgumentParser(description='Read live camera')
  parser.add_argument('--camera-source', '-c', type=int, default=0,
                      help='Specify the source of the camera, 0 as default. 1/2/3 are typical values')
  parser.add_argument('--base-dir', '-d', default='.',
                      help='Specify the path to store the data. Default to the current folder')
  parser.add_argument('--mode', type=str, required=True, help="Data collection for intrinsic or extrinsic calibration")
  parser.add_argument('--ethernet', type=str, required=False, help="Choose ethernet port on pc")

  args = parser.parse_args()
  
  if not args.mode in ["intrinsic", "extrinsic"]:
    raise ArgumentTypeError(None, "Must choose mode from intrinsic and extrinsic.")
  if args.mode == "extrinsic" and args.ethernet is None:
    raise ArgumentTypeError(None, "Must specify ethernet port while doing extrinsic calibration collection.")

  camera = cv2.VideoCapture(args.camera_source)

  image_folder = os.path.join(args.base_dir, 'camera_{}'.format(args.mode))
  if not os.path.exists(image_folder):
    os.mkdir(image_folder)
  # Create window with freedom of dimensions
  cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
  
  img_cnt = 0

  while True:
    ret, frame = camera.read()
    if not ret:
      print("Failed to grab image")
      break
    
    cv2.imshow("Live Camera", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
      # ESC Pressed
      print("ESC")
      break
    elif k % 256 == 32:
      # SPACE Pressed
      image_path = os.path.join(image_folder, "{}.png".format(img_cnt))
      cv2.imwrite(image_path, frame)
      print("Writing image to path: ", image_path)

      if args.mode == "extrinsic":
        pcap_path = os.path.join(image_folder, "{}.pcap".format(img_cnt))
        externalWiresharkCollection(pcap_path, args.ethernet, 10)
        print("Writing point cloud pcap to path: ", pcap_path)

      img_cnt += 1

  camera.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
