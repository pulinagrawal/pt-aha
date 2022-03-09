from os.path import join
from typing import List, Tuple
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, list_dir, list_files
import numpy as np
import imageio

class OmniglotAlphabet(Dataset):
    """ Modified from `https://pytorch.org/vision/stable/_modules/torchvision/datasets/omniglot.html`
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://raw.githubusercontent.com/brendenlake/omniglot/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718'
    }

    def __init__(self, root, alphabet, use_alphabet, own_alphabet=False, variation=False, writer_idx=1,
                 transform=None, target_transform=None, download=False):
        self.root = root
        self.alphabet = alphabet  # The alphabet of interest.
        self.own_alphabet = own_alphabet
        self.use_alphabet = use_alphabet  # Indicates wheather to only use this alphabet or use the rest of the alphabets.
        self.variation = variation
        self.writer_idx = writer_idx
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = join(self.root, self._get_target_folder())

        if self.use_alphabet: # Only the characters in the given alphabet.
          self.characters = [join(self.alphabet, c) for c in list_dir(join(self.target_folder, self.alphabet))]
          self.characters = sorted(self.characters)
          self.character_images = [[(image, idx) for image in sorted(list_files(join(self.target_folder, character), '.png'))]
                                  for idx, character in enumerate(self.characters)]
          if self.variation:
            self.flat_character_images = [[character[character_idx] for character_idx in range(0, 20)] for character
                                            in self.character_images]
          else:
            self.flat_character_images = [character[self.writer_idx] for character in self.character_images]
        else: # All characters except from those in the alphabet specified.
          self.alphabets = list_dir(self.target_folder)
          self.alphabets.remove(str(self.alphabet))
          self.characters: List[str] = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                           for a in self.alphabets], [])
          self.character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
                                  for idx, character in enumerate(self.characters)]
          self.flat_character_images = sum(self.character_images, [])

    def __len__(self):
        return len(self.flat_character_images)

    def __getitem__(self, index: int):

        if self.variation:
            character_images = self.flat_character_images[index]
            rand_character = np.random.randint(20)
            image_name, character_class = character_images[rand_character]
        else:
            image_name, character_class = self.flat_character_images[index]

        image_path = join(self.target_folder, self.characters[character_class], image_name)
        image = imageio.imread(image_path)

        # Convert to float values in [0, 1]
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _check_integrity(self):
        if not self.own_alphabet:
            zip_filename = self._get_target_folder()
            if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
                return False
            return True
        else:
            return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        filename = self._get_target_folder()
        zip_filename = filename + '.zip'
        url = self.download_url_prefix + '/' + zip_filename
        download_and_extract_archive(url, self.root, filename=zip_filename, md5=self.zips_md5[filename])


    def _get_target_folder(self):
        if self.own_alphabet:
            return 'own_alphabets'
        else:
            return 'images_background'


