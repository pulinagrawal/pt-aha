
import os

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.datasets.utils import check_integrity

import numpy as np

import imageio
from scipy import ndimage

from pathlib import Path
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
import shutil
import zipfile

class ImageTransformation:
  """Transform images by resizing and centring"""

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


class OneShotDataset(Dataset):
  """One-shot dataset."""
  fname_label = 'class_labels.txt'

  def __init__(self, root, dataset, num_runs=20, mode='train', transform=None, target_transform=None, download=False):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.dataset = dataset
    self.root = root
    self.num_runs = num_runs
    self.mode = mode
    self.transform = transform
    self.target_transform = target_transform

    if dataset == 'cifar100':
      self.folder = 'cifar-100-batches-py'
      self.file_id = '1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI'
      self.split = 'bertinetto'

    elif dataset == 'omniglot':
      self.folder = 'omniglot-batches-py'
      self.file_id = '10ml4OJRc13pl5Ms3mm2VyscyTj94c87O'
      self.split = 'vinyals'

    elif dataset == 'miniimagenet':
      self.folder = 'miniimagenet-batches-py'
      self.file_id = '1R6dA6QGEW-lmiNkitCwK4IkAbl4uT3y3'
      self.split = 'ravi-larochelle'

    else:
      raise RuntimeError('No dataset specified')

    self.root = os.path.join(root, self.folder)
    self.target_folder = self._get_target_folder()

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

    # Convert to float values in [0, 1
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())

    if self.transform:
      image = self.transform(image)

    if self.target_transform:
      label = self.target_transform(label)

    return image, label

  def _check_integrity(self):
    zip_filename = self._get_target_folder()
    if not check_integrity(os.path.join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
      return False
    return True

  def download(self):
    # this method downloads the CIFAR-100 few-shot dataset

    if self._check_integrity():
      print('Files already downloaded and verified')
      return

    zip_filename = self._get_target_folder() + '.zip'

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join("./data/auth.json")

    service = build('drive', 'v3')
    request = service.files().get_media(fileId=self.file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False

    while done is False:
      status, done = downloader.next_chunk()
      print("Download %d%%." % int(status.progress() * 100))

    print("Download completed")

    Path(self.root).mkdir(parents=True, exist_ok=True)

    # save
    fh.seek(0)
    with open(os.path.join(self.root, zip_filename), 'wb') as f:
      shutil.copyfileobj(fh, f)

    # unzip
    with zipfile.ZipFile(os.path.join(self.root, zip_filename)) as zip_ref:
      zip_ref.extractall(self.root)

  def get_filenames_and_labels(self):
    filenames = []
    labels = []

    for r in range(1, self.num_runs + 1):
      rs = str(r)
      if len(rs) == 1:
        rs = '0' + rs

      target_path = os.path.join(self.root, self.target_folder)
      # run_path = os.path.join(target_path, run_folder)
      '''
      with open(os.path.join(target_path, run_folder, self.fname_label)) as f:
        content = f.read().splitlines()
      pairs = [line.split() for line in content]

      test_files = [pair[0] for pair in pairs]
      train_files = [pair[1] for pair in pairs]

      train_labels = list(range(self.num_runs))
      test_labels = copy.copy(train_labels)      # same labels as train, because we'll read them in this order

      test_files = [os.path.join(target_path, file) for file in test_files]
      train_files = [os.path.join(target_path, file) for file in train_files]
'''

      split_dir = os.path.join(target_path, 'splits', self.split)

      if self.mode == 'train':
        split_file = 'train.txt'
        '''
        filenames.extend(train_files)
        labels.extend(train_labels)
'''
      elif self.mode == 'test':
        split_file = 'test.txt'
        '''
        filenames.extend(test_files)
        labels.extend(test_labels)
'''
      elif self.mode == 'val':
        split_file = 'val.txt'

        '''
        filenames.extend(val_files)
        labels.extend(val_labels)
        '''

      class_names = []
      with open(os.path.join(split_dir, split_file), 'r') as f:
        for class_name in f.readlines():
          class_names.append(class_name.rstrip('\n'))

    return filenames, labels

  def _get_target_folder(self):
    return self.dataset


"""
with open('data/cifar-10-batches-py/test_batch', 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

print(dict)
"""



