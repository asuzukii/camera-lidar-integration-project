import sys
import inspect
import cv2

from ImageProcessing import *


class IDS_UI5140:
    def __init__(self) -> None:
        # self.resolution = (1280, 1024)
        self.resolution = (2048, 1536)
        self.name = "IDS_UI5140"
        self.connection = "Ethernet"

    def postProcessing(self, frame):
        frame = cv2.flip(frame, 0)
        frame = gammaCorrection(frame, 2.2)
        frame = whiteBlance(frame, 0)
        return frame


class ELP:
    def __init__(self) -> None:
        self.resolution = (640, 480)
        self.name = "ELP"
        self.connection = "USB"


current_module = sys.modules[__name__]
CAMERA_MODELS = []
for name, obj in inspect.getmembers(current_module):
    if inspect.isclass(obj):
        CAMERA_MODELS.append(obj.__name__)
