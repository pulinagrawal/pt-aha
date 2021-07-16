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

  num_runs = -1
  fname_label = 'class_labels.txt'

  folder = 'omniglot_instance'
  download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
  zips_md5 = {
      'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
  }

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, batch_size=None):
    super(OmniglotInstanceDataset).__init__(root, train, transform, target_transform, download)

    self.batch_size = batch_size

  def __len__(self):
    return len(self.show_filenames)

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

  def _create_test_sets(self):
    """
    The order of samples is such that they are divided into batches,
    and always have one of each of the first `batch_size` classes from `test_show_classes`
    """

    # assert that batch_size <= test_classes
    # assert that batches <= len(labels) / batch_size

    # 1) Get full list of possible samples, filename and label
    # ------------------------------------------------------------------------------------
    images_folder = self._download(self._directory, 'images_background')
    files, labels = self._filenames_and_labels(images_folder)

    # filter the filenames, labels by the list in the_classes
    the_classes = self.get_classes_by_superclass(self._test_classes)

    files_filtered = []
    labels_filtered = []
    for file, label in zip(files, labels):
      if label in the_classes:
        files_filtered.append(file)
        labels_filtered.append(label)
    files = files_filtered
    labels = labels_filtered

    # 2) Sort the full list 'labels' into batches of unique classes
    # ------------------------------------------------------------------------------------
    self._dataset_show = []
    self._dataset_match = []

    # first shuffle the order
    dataset = list(zip(files, labels))
    np.random.shuffle(dataset)
    files, labels = map(list, zip(*dataset))

    # then repeatedly sample with removal, assembling all batches
    data_show = []
    data_match = []

    end_batches = False
    batch_num = -1
    while not end_batches:
      batch_num += 1

      if batch_num >= 20:
        break

      # build a new batch
      batch_labels = []
      batches_labels = []
      batch_label_index = -1
      batch_label = ''
      for i, sample in enumerate(range(self._batch_size)):
        # get the next index
        # ----------------------------

        index = -1
        # first item in batch, sample a random label that has not been chosen previously
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

        # from then on, choose another exemplar from the same class
        else:
          # select same class for a 'match' sample
          if batch_label in labels:
            index = labels.index(batch_label)

        logging.debug("==================     ----> Batch={}, index={}".format(batch_num, index))

        # detect reaching the end of the dataset i.e. not able to assemble a new batch
        if index == -1:
          logging.info('Not able to find a unique class to assemble a new batch, '
                        'on batch={0}, sample={1}'.format(batch_num, sample))
          end_batches = True
          break

        # add to the datasets
        file = files.pop(index)
        label = labels.pop(index)
        data_show.append([file, label])
        data_match.append([file, label])

    # convert from array of pairs, to pair of arrays
    self.show_filenames, self.show_labels = map(list, zip(*data_show))
    self.match_filenames, self.match_labels = map(list, zip(*data_match))

  def get_classes_by_superclass(self, superclasses, proportion=1.0):
    """
    Retrieves a proportion of classes belonging to a particular superclass, defaults to retrieving all classes
    i.e. proportion=1.0.

    Arguments:
      superclasses: A single or list of the names of superclasses, or a single name of a superclass.
      proportion: A float that indicates the proportion of sub-classes to retrieve (default=1.0)
    """
    if not self.CLASS_MAP:
      raise ValueError('Superclass to class mapping (CLASS_MAP) is not populated yet.')

    def filter_classes(classes, proportion, do_shuffle=True):
      """Filters the list of classes by retrieving a proportion of shuffled classes."""
      if do_shuffle:
        shuffle(classes)
      num_classes = math.ceil(len(classes) * float(proportion))
      return classes[:num_classes]

    classes = []
    if superclasses is None or (isinstance(superclasses, list) and len(superclasses) == 0):
      for superclass in self.CLASS_MAP.keys():
        subclasses = filter_classes(self.CLASS_MAP[superclass], proportion)
        classes.extend(subclasses)
    elif isinstance(superclasses, list):
      for superclass in superclasses:
        subclasses = filter_classes(self.CLASS_MAP[superclass], proportion)
        classes.extend(subclasses)
    else:   # string - single superclass specified
      classes = filter_classes(self.CLASS_MAP[superclasses], proportion)

    return classes

  def _filenames_and_labels(self, image_folder):
    """Get the image filename and label for each Omniglot character."""
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
    #   A 4D uint8 numpy array [index, y, x, depth].
    idx = 0
    filename_arr = []
    label_arr = np.zeros([num_images], dtype=np.int32)

    for path in character_folders:
      if os.path.isdir(path):
        for file in os.listdir(path):
          filename_arr.append(os.path.join(path, file))
          label_arr[idx] = file.split('_')[0]
          idx += 1

    return filename_arr, label_arr

  def _get_target_folder(self):
    return 'images_evaluation'

  def _get_phase_folder(self):
    return 'images_evaluation'

