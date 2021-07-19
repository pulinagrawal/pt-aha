"""OmniglotInstanceDataset class."""

import os
import math
import copy
import logging

from random import shuffle

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.datasets.utils import download_and_extract_archive, check_integrity

import numpy as np

import imageio
from scipy import ndimage

from datasets.omniglot_one_shot_dataset import OmniglotOneShotDataset


class OmniglotInstanceDataset(OmniglotOneShotDataset):
  """Omniglot Instance dataset."""

  # Mapping of alphabets (superclasses) to characters (classes) populated when dataset is loaded
  CLASS_MAP = {}

  folder = 'omniglot_instance'
  download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
  zips_md5 = {
      'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
  }

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, batch_size=None,
               show_filenames=[], show_labels=[], match_filenames=[], match_labels=[]):
    super(OmniglotInstanceDataset, self).__init__(root, train, transform, target_transform, download)

    self.batch_size = batch_size
    self.show_filenames = show_filenames
    self.show_labels = show_labels
    self.match_filenames = match_filenames
    self.match_labels = match_labels

    if not all([self.show_filenames, self.show_labels, self.match_filenames, self.match_labels]):
      print('Generating show and match sets...')
      self._generate_show_and_match_sets()

  def __len__(self):
    return len(self.show_filenames)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    if self.train:
      label = self.show_labels[idx]
      image_path = self.show_filenames[idx]
    else:
      label = self.match_labels[idx]
      image_path = self.match_filenames[idx]

    image = imageio.imread(image_path)

    # Convert to float values in [0, 1
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())

    if self.transform:
      image = self.transform(image)

    if self.target_transform:
      label = self.target_transform(label)

    return image, label

  def _generate_show_and_match_sets(self):
    """
    The order of samples is such that they are divided into batches,
    and always have one of each of the first `batch_size` classes from `test_show_classes`
    """

    use_real_labels = False

    self._dataset_show = []
    self._dataset_match = []

    # Shuffle the data
    dataset = list(zip(self.filenames, self.labels))
    np.random.shuffle(dataset)
    files, labels = map(list, zip(*dataset))

    data_show = []
    data_match = []
    end_batches = False
    batch_num = -1

    # Repeatedly sample with removal, assembling all batches
    while not end_batches:
      batch_num += 1

      if batch_num >= 20:
        break

      # Build a new batch
      batches_labels = []
      batch_label_index = -1
      batch_label = ''

      for i, sample in enumerate(range(self.batch_size)):
        # Get the next index
        # ----------------------------
        index = -1

        # First item in batch - Sample a random label that has not been chosen previously
        if batch_label_index == -1:
          # select first sample that is not in all batches so far (we want each batch to be a unique class)
          for idx, label in enumerate(labels):
            if label not in batches_labels:
              batch_label_index = idx
              batch_label = label
              batches_labels.append(batch_label)  # remember which labels we added to all batches
              index = idx
              break

          logging.debug("====================== Batch={}, label={}".format(batch_num, batch_label))

        # From then on, choose another exemplar from the same class
        else:
          # select same class for a 'match' sample
          if batch_label in labels:
            index = labels.index(batch_label)

        logging.debug("==================     ----> Batch={}, index={}".format(batch_num, index))

        # Detect reaching the end of the dataset i.e. not able to assemble a new batch
        if index == -1:
          logging.info('Not able to find a unique class to assemble a new batch, '
                        'on batch={0}, sample={1}'.format(batch_num, sample))
          end_batches = True
          break

        # Add to the datasets
        file = files.pop(index)
        label = labels.pop(index)

        # Replace real label with sample index
        # Note: this would be most similar to the Lake runs
        presented_label = label
        if not use_real_labels:
          presented_label = i

        data_show.append([file, presented_label])
        data_match.append([file, presented_label])

    # Convert from array of pairs, to pair of arrays
    self.show_filenames, self.show_labels = map(list, zip(*data_show))
    self.match_filenames, self.match_labels = map(list, zip(*data_match))

  def get_filenames_and_labels(self):
    """Get the image filename and label for each Omniglot character."""
    image_folder = os.path.join(self.root, self.target_folder, self.phase_folder)

    # Compute list of characters (each is a folder full of images)
    character_folders = []
    for family in os.listdir(image_folder):
      if os.path.isdir(os.path.join(image_folder, family)):
        append_characters = False
        if family not in self.CLASS_MAP:
          self.CLASS_MAP[family] = []
          append_characters = True
        for character in os.listdir(os.path.join(image_folder, family)):
          character_folder = os.path.join(image_folder, family, character)
          if append_characters and os.path.isdir(character_folder):
            character_file = os.listdir(character_folder)[0]
            character_label = int(character_file.split('_')[0])
            self.CLASS_MAP[family].append(character_label)
          character_folders.append(character_folder)
      else:
        logging.warning('Path to alphabet is not a directory: %s', os.path.join(image_folder, family))

    # Count number of images
    num_images = 0
    for path in character_folders:
      if os.path.isdir(path):
        for file in os.listdir(path):
          num_images += 1

    # Put them in one big array, and one for labels
    # A 4D uint8 numpy array [index, y, x, depth].
    idx = 0
    filename_arr = []
    label_arr = []

    for path in character_folders:
      if os.path.isdir(path):
        for file in os.listdir(path):
          filename = os.path.join(path, file)
          label_string = file.split('_')[0]
          label = int(label_string)

          filename_arr.append(filename)
          label_arr.append(label)
          idx += 1

    return filename_arr, label_arr

  def _get_target_folder(self):
    return 'images_evaluation'

  def _get_phase_folder(self):
    return 'images_evaluation'

