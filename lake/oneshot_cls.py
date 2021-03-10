"""oneshot.py"""

import os
import json
import argparse
import logging

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

from cls_module.cls import CLS

import utils

from omniglot_one_shot_dataset import OmniglotTransformation, OmniglotOneShotDataset
from oneshot_metrics import OneshotMetrics

LOG_EVERY = 20


def main():
  parser = argparse.ArgumentParser(description='Complementary Learning System: One-shot Learning Experiments')
  parser.add_argument('-c', '--config', nargs="?", type=str, default='./definitions/aha_config.json',
                      help='Configuration file for experiments.')
  parser.add_argument('-l', '--logging', nargs="?", type=str, default='warning',
                      help='Logging level.')

  args = parser.parse_args()

  logging_level = getattr(logging, args.logging.upper(), None)
  logging.basicConfig(level=logging_level)

  with open(args.config) as config_file:
    config = json.load(config_file)

  utils.set_seed(config['seed'])

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  summary_dir = utils.get_summary_dir()

  writer = SummaryWriter(log_dir=summary_dir)

  image_tfms = transforms.Compose([
      transforms.ToTensor(),
      OmniglotTransformation(resize_factor=config['image_resize_factor'])
  ])

  image_shape = config['image_shape']
  pretrained_model_path = config.get('pretrained_model_path', None)

  # Pretraining
  # ---------------------------------------------------------------------------
  if not pretrained_model_path:
    background_loader = torch.utils.data.DataLoader(
        datasets.Omniglot('./data', background=True, download=True, transform=image_tfms),
        batch_size=config['pretrain_batch_size'], shuffle=True)

    model = CLS(image_shape, config, device=device, writer=writer).to(device)

    # Pre-train the model
    for epoch in range(1, config['pretrain_epochs'] + 1):
      model.train()

      for batch_idx, (data, target) in enumerate(background_loader):
        data, target = data.to(device), target.to(device)

        losses, _ = model(data, labels=target, mode='pretrain')
        pretrain_loss = losses['ltm']['memory']['loss'].item()

        if batch_idx % LOG_EVERY == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(background_loader.dataset),
              100. * batch_idx / len(background_loader), pretrain_loss))

    pretrained_model_path = os.path.join(summary_dir, 'pretrained_model_' + str(epoch) + '.pt')

    print('Saving model to:', pretrained_model_path)
    torch.save(model.state_dict(), pretrained_model_path)

  # Study and Recall
  # ---------------------------------------------------------------------------

  # Prepare data loaders
  study_loader = torch.utils.data.DataLoader(
      OmniglotOneShotDataset('./data', train=True, download=True,
                             transform=image_tfms, target_transform=None),
      batch_size=config['study_batch_size'], shuffle=False)

  recall_loader = torch.utils.data.DataLoader(
      OmniglotOneShotDataset('./data', train=False, download=True,
                             transform=image_tfms, target_transform=None),
      batch_size=config['study_batch_size'], shuffle=False)

  assert len(study_loader) == len(recall_loader)

  oneshot_dataset = enumerate(zip(study_loader, recall_loader))

  # Initialise metrics
  oneshot_metrics = OneshotMetrics()

  # Load the pretrained model
  model = CLS(image_shape, config, device=device, writer=writer).to(device)

  for idx, ((study_data, study_target), (recall_data, recall_target)) in oneshot_dataset:
    study_data = study_data.to(device)
    study_target = torch.from_numpy(np.array(study_target)).to(device)
    recall_data = recall_data.to(device)
    recall_target = torch.from_numpy(np.array(recall_target)).to(device)

    # Reset to saved model
    model.load_state_dict(torch.load(pretrained_model_path))
    model.reset()

    # Study
    # --------------------------------------------------------------------------
    model.train()
    for step in range(config['study_steps']):
      model(study_data, study_target, mode='study')

      if step % LOG_EVERY == 0:
        print('Run #{}: [{}/{}]'.format(idx, step, config['study_steps']))

    # Recall
    # --------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
      model(recall_data, recall_target, mode='recall')

      if 'metrics' in config and config['metrics']:
        metrics_config = config['metrics']

        metrics_len = [len(value) for key, value in metrics_config.items()]
        assert all(x == metrics_len[0] for x in metrics_len), 'Mismatch in metrics config'

        for i in range(metrics_len[0]):
          primary_feature = utils.find_json_value(metrics_config['primary_feature_names'][i], model.features)
          primary_label = utils.find_json_value(metrics_config['primary_label_names'][i], model.features)
          secondary_feature = utils.find_json_value(metrics_config['secondary_feature_names'][i], model.features)
          secondary_label = utils.find_json_value(metrics_config['secondary_label_names'][i], model.features)
          comparison_type = metrics_config['comparison_types'][i]
          prefix = metrics_config['prefixes'][i]

          oneshot_metrics.compare(prefix,
                                  primary_feature, primary_label,
                                  secondary_feature, secondary_label,
                                  comparison_type=comparison_type)

      # PR Accuracy
      oneshot_metrics.compare('pr',
                              None, model.features['study']['stm_pr'],
                              None, model.features['recall']['stm_pr'],
                              comparison_type='accuracy')

      oneshot_metrics.report()

  oneshot_metrics.report_averages()

  writer.flush()
  writer.close()

if __name__ == '__main__':
  main()
