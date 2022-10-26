import ctypes
import numpy as np

path = "../extrinsic_calibration_cpp/build/lib/libextrinsic_calibration.so"

lib = ctypes.CDLL(path)

"""
struct PointArray {
    float* x;
    float* y;
    float* z;
    float* intensity;
    uint8_t* channel_id;
    bool* valid;
    bool* saturated;
    size_t length;
};

struct CornerArray {
    float* x;
    float* y;
    float* z;
    int* corr;
};

#ifdef __cplusplus
extern "C" {
#endif

void CornerExtractionExport(PointArray* pa, size_t cloudLength, CornerArray* ca,
                            float* kernel, int kernelSize, int index,
                            float square_size, size_t num_vertical,
                            size_t num_horizontal, float height, float width);

void PreprocessExport(PointArray* pa, float x_min, float x_max, float y_min,
                      float y_max, float z_min, float z_max,
                      float intensity_min, float intensity_max);

"""
PreprocessExport = lib.PreprocessExport
CornerExtractionExport = lib.CornerExtractionExport
GetBufferSize = lib.GetImageBufferSize
CopyImage = lib.CopyImage


class PointArray(ctypes.Structure):
  _fields_ = [
    ("x", ctypes.POINTER(ctypes.c_float)),
    ("y", ctypes.POINTER(ctypes.c_float)),
    ("z", ctypes.POINTER(ctypes.c_float)),
    ("intensity", ctypes.POINTER(ctypes.c_float)),
    ("channel_id", ctypes.POINTER(ctypes.c_uint8)),
    ("valid", ctypes.POINTER(ctypes.c_bool)),
    ("saturated", ctypes.POINTER(ctypes.c_bool)),
    ("length", ctypes.c_size_t)
  ]

  @classmethod
  def PointCloudPackage(self, data):
    x = data[:, 0].astype(np.float32)
    y = data[:, 1].astype(np.float32)
    z = data[:, 2].astype(np.float32)
    r = data[:, 3].astype(np.float32)
    v = data[:, 4].astype(np.bool8)
    s = data[:, 5].astype(np.bool8)
    c = data[:, 6].astype(np.uint8)

    pa = PointArray()

    pa.x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pa.y = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pa.z = z.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pa.intensity = r.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pa.channel_id = c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    pa.saturated = s.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
    pa.valid = v.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
    pa.length = len(data)

    return pa


class CornerArray(ctypes.Structure):
  _fields_ = [
    ("x", ctypes.POINTER(ctypes.c_float)),
    ("y", ctypes.POINTER(ctypes.c_float)),
    ("z", ctypes.POINTER(ctypes.c_float)),
    ("corr", ctypes.POINTER(ctypes.c_int32))
  ]

  @classmethod
  def CornerPackage(self):
    pass

def CornerExtractionCpp(input: np.ndarray, kernel: np.ndarray, index: int, 
    square_size: float, num_vertical: int, num_horizontal: int, height: float, 
    width: float):
  data = np.copy(input)

  x = data[:, 0].astype(np.float32)
  y = data[:, 1].astype(np.float32)
  z = data[:, 2].astype(np.float32)
  r = data[:, 3].astype(np.float32)
  v = data[:, 4].astype(np.bool8)
  s = data[:, 5].astype(np.bool8)
  c = data[:, 6].astype(np.uint8)

  pa = PointArray()

  pa.x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  pa.y = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  pa.z = z.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  pa.intensity = r.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  pa.channel_id = c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
  pa.saturated = s.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
  pa.valid = v.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
  pa.length = len(data)

  ca = CornerArray()

  corners_x = np.zeros((num_vertical-1) * (num_horizontal-1), dtype=np.float32)
  corners_y = np.zeros((num_vertical-1) * (num_horizontal-1), dtype=np.float32)
  corners_z = np.zeros((num_vertical-1) * (num_horizontal-1), dtype=np.float32)
  corr = np.zeros((num_vertical-1) * (num_horizontal-1), dtype=np.int32)

  ca.x = corners_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  ca.y = corners_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  ca.z = corners_z.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  ca.corr = corr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

  assert kernel.shape[0] == kernel.shape[1] and len(kernel.shape) == 2
  kernelSize = kernel.shape[0]

  kernel_flatten = kernel.flatten().astype(np.float32).copy()
  kernelPointer = kernel_flatten.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

  CornerExtractionExport(ctypes.byref(pa), ctypes.byref(ca), kernelPointer, ctypes.c_int(kernelSize), 
      ctypes.c_int(index), ctypes.c_float(square_size), 
      ctypes.c_size_t(num_vertical), ctypes.c_size_t(num_horizontal), 
      ctypes.c_float(height), ctypes.c_float(width))
  width = ctypes.c_size_t()
  height = ctypes.c_size_t()
  GetBufferSize(ctypes.byref(width), ctypes.byref(height))

  image = np.zeros(np.ctypeslib.as_array(width)* np.ctypeslib.as_array(height), dtype=np.float32)

  CopyImage(image.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

  image = np.reshape(image, [np.ctypeslib.as_array(height), np.ctypeslib.as_array(width)])
  return image

def PreprocessCpp(input: np.ndarray, x_min, x_max, y_min, y_max,
                  z_min, z_max, intensity_min, intensity_max):

  data = np.copy(input)

  x = data[:, 0].astype(np.float32)
  y = data[:, 1].astype(np.float32)
  z = data[:, 2].astype(np.float32)
  r = data[:, 3].astype(np.float32)
  v = data[:, 4].astype(np.bool8)
  s = data[:, 5].astype(np.bool8)
  c = data[:, 6].astype(np.uint8)

  pa = PointArray()

  pa.x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  pa.y = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  pa.z = z.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  pa.intensity = r.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  pa.channel_id = c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
  pa.saturated = s.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
  pa.valid = v.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
  pa.length = len(data)

  PreprocessExport(ctypes.byref(pa), 
                   ctypes.c_float(x_min), ctypes.c_float(x_max), 
                   ctypes.c_float(y_min), ctypes.c_float(y_max),
                   ctypes.c_float(z_min), ctypes.c_float(z_max), 
                   ctypes.c_float(intensity_min), ctypes.c_float(intensity_max))

  data = np.stack([x, y, z, r, v, s, c], 1)
  data = data[:pa.length, :]

  return data
