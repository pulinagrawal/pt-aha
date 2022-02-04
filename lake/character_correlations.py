import os
import argparse
import logging
import json
import torch.nn
import glob
from scipy import stats
from cls_module.cls import CLS
from datasets.omniglot_one_shot_dataset import OmniglotTransformation
from datasets.omniglot_per_alphabet_dataset import OmniglotAlphabet
from torchvision import transforms

#Trained without N_Ko alphabet
pretrained_model_path = "./runs/pairs_structure/episodic/20220112-234052/6517/"

#Trained without N_Ko alphabet
#pretrained_model_path = "./runs/associative_inference/static/20220131-140733/1150/"

parser = argparse.ArgumentParser(description='Pair structure. Replicate of Schapiro')
parser.add_argument('-c', '--config', nargs="?", type=str, default='./definitions/aha_config_Schapiro_episodic.json',
                        help='Configuration file for experiments.')

parser.add_argument('-l', '--logging', nargs="?", type=str, default='warning',
                        help='Logging level.')
args = parser.parse_args()
logging_level = getattr(logging, args.logging.upper(), None)
logging.basicConfig(level=logging_level)

with open(args.config) as config_file:
    config = json.load(config_file)

alphabet_name = 'N_Ko_2'
writer_idx = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_tfms = transforms.Compose([
    transforms.ToTensor(),
    OmniglotTransformation(resize_factor=config['image_resize_factor'])])

model = CLS(config['image_shape'], config, device=device, writer=config['writer_idx_study'], output_shape=config['pairs_shape']).to(device)
model_path = os.path.join(pretrained_model_path, 'pretrained_model_*')
print(model_path)
 # Find the latest file in the directory
latest = max(glob.glob(model_path), key=os.path.getctime)
trained_model_path = latest

# Load only the LTM part
pretrained_dict = torch.load(trained_model_path)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('ltm')}
model.load_state_dict(pretrained_dict)


alphabet = OmniglotAlphabet('./data', alphabet_name, use_alphabet=True, writer_idx=writer_idx,
                            transform=image_tfms, target_transform=None)

single_characters = [alphabet[a][0] for a in range(0, len(alphabet))]
single_characters = torch.stack(single_characters)
characters_target = list(range(single_characters.size()[0]))
characters_target = torch.tensor(characters_target, dtype=torch.long, device=device)
_, output_characters = model(single_characters, characters_target, mode='extractor')
output_characters = output_characters['ltm']['memory']['output'].detach()

correlation_recall = torch.flatten(output_characters, start_dim=1)

correlation_recall = [[stats.pearsonr(a, b)[0] for a in correlation_recall] for b in
                      correlation_recall]
correlation_recall = torch.tensor(correlation_recall).numpy()


