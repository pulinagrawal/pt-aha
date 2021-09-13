"""ImageRetriever class."""

import os
import copy

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.datasets.utils import download_and_extract_archive, check_integrity

import numpy as np

import imageio
from scipy import ndimage


class ImageRetriever(Dataset):
  """Face Landmarks dataset."""

  num_runs = 3
  fname_label = 'class_labels.txt'

  folder = 'Pair_structure'
  download_url_prefix = 'https://github.com/brendenlake/omniglot/tree/master/python'
  zips_md5 = {
      'all_runs': '68d2efa1b9178cc56df9314c21c6e718'
  }

  def __init__(self, root, alphabet,  train=True, transform=None, target_transform=None, download=False):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.root = root
    self.alphabet = alphabet
    self.train = train
    self.transform = transform
    self.target_transform = target_transform

    self.root = os.path.join(root, self.folder)
    self.target_folder = self._get_target_folder()
    self.phase_folder = self._get_phase_folder()

    if download:
      self.download()

    self.filenames, self.labels = self.get_filenames_and_labels()

    if not self._check_integrity():
      raise RuntimeError('Dataset not found or corrupted.' +
                         ' You can use download=True to download it')

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    label = self.labels[idx]

    image_path = self.filenames[idx]
    image = imageio.imread(image_path)

    # Convert to float values in [0, 1]
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())

    if self.transform:
      image = self.transform(image)

    if self.target_transform:
      label = self.target_transform(label)

    return image, label

  def _check_integrity(self):
    zip_filename = self._get_target_folder()
    print(zip_filename)
    if not check_integrity(os.path.join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
      return False
    return True

  def download(self):
    if self._check_integrity():
      print('Files already downloaded and verified')
      return

    filename = self._get_target_folder()
    zip_filename = filename + '.zip'
    url = self.download_url_prefix + '/' + zip_filename
    download_and_extract_archive(url, self.root,
                                 extract_root=os.path.join(self.root, filename),
                                 filename=zip_filename, md5=self.zips_md5[filename])

  def get_filenames_and_labels(self):
    filenames = []
    labels = []

    for r in range(1, self.num_runs + 1):
      rs = str(r)
      if len(rs) == 1:
        rs = '0' + rs

      run_folder = 'run' + rs
      target_path = os.path.join(self.root, self.target_folder + '/' + self.alphabet)
      # run_path = os.path.join(target_path, run_folder)

      with open(os.path.join(target_path, run_folder, self.fname_label)) as f:
        content = f.read().splitlines()
      pairs = [line.split() for line in content]

      test_files = [pair[0] for pair in pairs]
      train_files = [pair[1] for pair in pairs]

      train_labels = list(range(self.num_runs))
      test_labels = copy.copy(train_labels)      # same labels as train, because we'll read them in this order

      test_files = [os.path.join(target_path, file) for file in test_files]
      train_files = [os.path.join(target_path, file) for file in train_files]

      if self.train:
        filenames.extend(train_files)
        labels.extend(train_labels)
      else:
        filenames.extend(test_files)
        labels.extend(test_labels)

    return filenames, labels

  def _get_target_folder(self):
    return 'images_background'

  def _get_phase_folder(self):
    return 'training' if self.train else 'test'
