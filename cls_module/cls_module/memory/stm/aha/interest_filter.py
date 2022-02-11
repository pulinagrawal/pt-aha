from random import gauss
import torch
import torch.nn as nn

import math
import numbers
import torch

from torch import nn
from torch.nn import functional as F

import torchvision

import cerenaut_pt_core.utils as utils

def gaussian_kernel(channels, kernel_size, sigma, dim=2):
  if isinstance(kernel_size, numbers.Number):
      kernel_size = [kernel_size] * dim
  if isinstance(sigma, numbers.Number):
      sigma = [sigma] * dim

  # The gaussian kernel is the product of the
  # gaussian function of each dimension.
  kernel = 1
  meshgrids = torch.meshgrid(
      [
          torch.arange(size, dtype=torch.float32)
          for size in kernel_size
      ]
  )
  for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
      mean = (size - 1) / 2
      kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)

  # Make sure sum of values in gaussian kernel equals 1.
  kernel = kernel / torch.sum(kernel)

  # Reshape to depthwise convolutional weight
  kernel = kernel.view(1, 1, *kernel.size())
  kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

  return kernel

def dog_kernel(channels, kernel_size, sigma, k):
  g1 = gaussian_kernel(channels, kernel_size, sigma)
  g2 = gaussian_kernel(channels, kernel_size, sigma  * k)
  return g1 - g2

class InterestFilter:
  def __init__(self):
    self.config = {
        "num_features": 15,
        "scale_range": [1],
        "pe_size": 15,
        "pe_std": 2.375,
        "nms_size": 5,
        "nms_stride": 1,
        "f_size": 7,
        "f_std": 1,
        "f_k": 1.6,
    }

  def __call__(self, image_tensor, conv_encodings):
    # get conv output at interest points
    fixation_mask = self.interest_function(image_tensor)    # input_image dims
    masked_encodings = conv_encodings * fixation_mask

    # smooth position of masked encodings
    smooth_k = gaussian_kernel(masked_encodings.shape[1], self.config['pe_size'], self.config['pe_std'])

    padded_masked_encodings = F.pad(masked_encodings, (2, 2, 2, 2), mode='reflect')
    positional_encodings = F.conv2d(padded_masked_encodings, weight=smooth_k, groups=masked_encodings.shape[1])
    positional_encodings = F.interpolate(positional_encodings, size=[masked_encodings.shape[2], masked_encodings.shape[3]])

    return fixation_mask, positional_encodings

  def interest_function(self, image_tensor):
    image_shape = image_tensor.shape
    channels = image_tensor.shape[1]
    image_shape_hw = [image_shape[2], image_shape[3]]

    # get kernels for later use
    dog_k = dog_kernel(channels, self.config['f_size'], self.config['f_std'], self.config['f_k'])

    def interest_core(positive=True):
      dog_kernel = dog_k
      if not positive:
        dog_kernel = -dog_k

      # analysis at multiple scales
      conv_sum = None
      for scale in self.config['scale_range']:

        # DoG kernel - edge and corner detection plus smoothing
        zoom_image_shape =  int(scale) * image_shape_hw
        image = F.interpolate(image_tensor, size=zoom_image_shape)

        padded_image = F.pad(image, (2, 2, 2, 2), mode='reflect')
        image = F.conv2d(padded_image, dog_kernel, stride=1, padding=0)
        image = F.interpolate(image_tensor, size=image_shape_hw)

        # non-maxima suppression (i.e. blend the peaks) at this scale
        image = self._non_maxima_suppression(image)

        # accumulate features at this scale
        if conv_sum is None:
          conv_sum = image
        else:
          conv_sum = conv_sum + image

      # sparse filtering
      k = self.config['num_features']
      fixations_mask_1d = utils.build_topk_mask(conv_sum, dim=-1, k=k)
      fixations_mask = torch.reshape(fixations_mask_1d, image_shape)

      return fixations_mask

    fixs_mask_pos = interest_core(positive=True)
    fixs_mask_neg = interest_core(positive=False)

    fixs_mask = fixs_mask_neg + fixs_mask_pos

    return fixs_mask

  def _non_maxima_suppression(self, image):
    use_smoothing = False
    pool_size = self.config['nms_size']
    stride = self.config['nms_stride']

    if use_smoothing:
      smooth_k = gaussian_kernel(image.shape[1], pool_size, self.config['nms_std'])
      image = F.conv2d(image, weight=smooth_k, strides=1, padding=0)
      return image

    pooled, indices = F.max_pool2d(image, kernel_size=pool_size, stride=stride, padding=0, return_indices=True)
    unpooled = F.max_unpool2d(pooled, kernel_size=pool_size, stride=stride, indices=indices)

    return unpooled
