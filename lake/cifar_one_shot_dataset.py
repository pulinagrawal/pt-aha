"""CifarOneShotDataset class."""

import os

from PIL import Image


import torchvision
from torch.utils.data import Dataset, DataLoader

from cifar_base import CIFAR100ClassDataset

import numpy as np

from scipy import ndimage

import torch


class CifarTransformation:
  """Transform Cifar images by resizing and centring"""

  def __init__(self, centre=True, invert=True, resize_factor=1.0):
    self.centre = centre
    self.resize_factor = resize_factor

  def __call__(self, x):
    # Resize
    if self.resize_factor != 1.0:
      height = int(self.resize_factor * x.shape[1])
      width = int(self.resize_factor * x.shape[2])

      x = torchvision.transforms.ToPILImage()(x)
      x = torchvision.transforms.functional.resize(x, size=[height, width])

      x = torchvision.transforms.functional.to_tensor(x)


    # Centre the image
    if self.centre:
      # NCHW => NHWC
      x = x.permute(1, 2, 0)

      # Compute centre
      centre = np.array([int(x.shape[0]) * 0.5, int(x.shape[1]) * 0.5])

      # Compute centre of mass
      centre_of_mass = ndimage.measurements.center_of_mass(x.numpy())
      centre_of_mass = np.array(centre_of_mass[:-1])

      # Compute translation
      translation = (centre - centre_of_mass).tolist()
      translation.reverse()

      # Apply transformation
      # NHWC => NCHW
      x = x.permute(2, 0, 1)
      x = torchvision.transforms.ToPILImage()(x)
      x = torchvision.transforms.functional.affine(x, 0, translation, scale=1.0, shear=0, resample=Image.BILINEAR)

      # Convert back to tensor
      x = torchvision.transforms.functional.to_tensor(x)

    return x


class CifarOneShotDataset(Dataset):
  """CIFAR one-shot dataset."""

  num_runs = 10
  fname_label = 'class_labels.txt'

  folder = 'cifar-100-batches-py'

  def __init__(self, root, mode='train', transform=None, target_transform=None, classes=None, download=False):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.root = os.path.join(root, self.folder)
    self.mode = mode

    self.transform = transform
    self.target_transform = target_transform

    self.target_folder = self._get_target_folder()

    self.dataset = CIFAR100ClassDataset(os.path.join(root, self.folder), meta_train=True, meta_val=False, meta_test=False,
                                        meta_split=None, transform=None, class_augmentations=None)

    self.classes = classes
    if self.classes is None:
      np.random.seed(71)
      self.classes = np.random.randint(low=0, high=64, size=20)  # array of 20 random integers to select classes


    self.images, self.labels = self.get_images_and_labels()

  def __len__(self):
    return len(self.labels)

  def get_images_and_labels(self):
    images = []
    labels = []

    num_runs = len(self.classes)
    dataset = self.dataset
    np.random.seed(63)

    for r in range(0, num_runs):
      for i in range(len(self.classes)):
        a_class = self.classes[i]

        # selection of image
        if self.mode == 'train':
          selection = np.random.randint(low=0, high=300)
        else:
          selection = np.random.randint(low=300, high=600)

        images.append(dataset[a_class][selection][0])  # PIL image
        #labels.append(dataset[a_class][selection][1])  # 'fine' class label
        labels.append(i) # classify by class number instead

    return images, labels

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    label = self.labels[idx]

    # convert PIL image to numpy array
    PIL_img = self.images[idx]
    image = np.array(PIL_img)

    # Convert to float values in [0, 1
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())

    if self.transform:
      image = self.transform(image)

    if self.target_transform:
      label = self.target_transform(label)

    return image, label

  def _get_target_folder(self):
    return 'cifar100'
