import os
from os.path import join
import shutil
import argparse
import logging
import glob
import json
from typing import List, Tuple
import torch.nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, list_dir, list_files
import numpy as np
import imageio
from torchvision import transforms
from datasets.omniglot_one_shot_dataset import OmniglotTransformation
from scipy import stats
from cls_module.cls import CLS

class ImageSet(Dataset):
    """ Modified from `https://pytorch.org/vision/stable/_modules/torchvision/datasets/omniglot.html`
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://raw.githubusercontent.com/brendenlake/omniglot/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718'
    }

    def __init__(self, root, index, transform=None, target_transform=None, download=False):
        self.root = root
        self.index = index
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = join(self.root, self._get_target_folder())

        self.alphabets = list_dir(join(self.target_folder))

        self.characters: List[str] = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                           for a in self.alphabets], [])
        self.characters = sorted(self.characters)

        self.character_images = [(list_files(join(self.target_folder, character), '.png')[self.index], idx)
                                     for idx, character in enumerate(self.characters)]



    def __len__(self):
        return len(self.character_images)

    def __getitem__(self, index: int):

        image_name, character_class = self.character_images[index]

        image_path = join(self.target_folder, self.characters[character_class], image_name)
        image = imageio.imread(image_path)

        # Convert to float values in [0, 1]
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class, image_path

    def _check_integrity(self):
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
            return False
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
        return 'images_background'


if __name__ == '__main__':
    score = 0.15
    set_name = 'Mix_0_15_015_test'
    hebbian = False

    image_tfms = transforms.Compose([
        transforms.ToTensor(),
        OmniglotTransformation(resize_factor=0.5)])
    set_size = 15
    test = ImageSet('./data',  index=0, transform=image_tfms, target_transform=None)

    best_Image_Set = []
    counter_image = 42
    best_Image_Set.append(test[counter_image])
    labels_best_set = [counter_image]
    images_path = [test[counter_image][2]]

    while len(best_Image_Set) < set_size:
        if test[counter_image+1][1] not in labels_best_set:
            tmp_image = test[counter_image+1]
            correlation = [stats.pearsonr(torch.flatten(a[0], start_dim=0),
                                          torch.flatten(tmp_image[0], start_dim=0))[0] for a in best_Image_Set]

            valid_correlation = np.product([a < score for a in correlation]) == 1

            if valid_correlation:
                best_Image_Set.append(tmp_image)
                labels_best_set.append(tmp_image[1])
                images_path.append(tmp_image[2])
                counter_image = counter_image + 1
                print(correlation)
            else:
                if counter_image + 1 == len(test)-1:
                    score = score + 0.02
                    counter_image = 0
                else:
                    counter_image = counter_image + 1

        else:
            counter_image = counter_image + 1

    for f in range(len(images_path)):
        name = images_path[f]
        if f < 9:
            label = '0' + str(f+1)
        else:
            label = str(f)
        tmp_path = './data/own_alphabets/' + set_name + '_simple/character' + label
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)
        shutil.copy(name, tmp_path)




    pretrained_model_path = "./runs/associative_inference/static/20220131-140733/1150/"
    parser = argparse.ArgumentParser(description='Replicate of Schapiro')
    if hebbian:
        parser.add_argument('-c', '--config', nargs="?", type=str,
                            default='./definitions/aha_config_Schapiro_hebb.json',
                            help='Configuration file for experiments.')
    else:
        parser.add_argument('-c', '--config', nargs="?", type=str, default='./definitions/aha_config_Schapiro.json',
                            help='Configuration file for experiments.')

    parser.add_argument('-l', '--logging', nargs="?", type=str, default='warning',
                        help='Logging level.')

    args = parser.parse_args()

    logging_level = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=logging_level)

    with open(args.config) as config_file:
        config = json.load(config_file)

    summary_dir = './summaries/' + set_name
    writer = SummaryWriter(log_dir=summary_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CLS(config['image_shape'], config, device=device, writer=writer, output_shape=config['pairs_shape']).to(
        device)

    model_path = os.path.join(pretrained_model_path, 'pretrained_model_*')
    # Find the latest file in the directory
    latest = max(glob.glob(model_path), key=os.path.getctime)

    trained_model_path = latest

    # Load only the LTM part
    pretrained_dict = torch.load(trained_model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('ltm')}
    model.load_state_dict(pretrained_dict)

    best_Image_Set = []
    counter_image = 42


    data = [test[a][0] for a in range(len(test))]
    data_tensor = torch.stack(data, 0)
    targets = [test[a][1] for a in range(len(test))]
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    _, data = model(data_tensor, targets, mode='extractor')
    data = data['ltm']['memory']['output'].detach()
    best_Image_Set.append(data[counter_image])
    labels_best_set = [counter_image]
    images_path = [test[counter_image][2]]

    while len(best_Image_Set) < set_size:

        if test[counter_image+1][1] not in labels_best_set:

            tmp_image = data[counter_image+1]

            correlation = [stats.pearsonr(torch.flatten(a, start_dim=0),
                                          torch.flatten(tmp_image, start_dim=0))[0] for a in best_Image_Set]

            valid_correlation = np.product([a < score for a in correlation]) == 1

            if valid_correlation:
                best_Image_Set.append(tmp_image)
                labels_best_set.append(test[counter_image+1][1])
                images_path.append(test[counter_image+1][2])
                counter_image = counter_image + 1
                print(correlation)
            else:
                if counter_image + 1 == len(test)-1:
                    score = score + 0.02
                    counter_image = 0
                else:
                    counter_image = counter_image + 1

        else:
            counter_image = counter_image + 1

    for f in range(len(images_path)):
        name = images_path[f]
        if f < 9:
            label = '0' + str(f+1)
        else:
            label = str(f)
        tmp_path = './data/own_alphabets/' + set_name + '/character' + label
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)
        shutil.copy(name, tmp_path)