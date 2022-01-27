"""oneshot.py"""

import os
import json
import argparse
import logging
import glob
from tqdm import tqdm

import numpy as np
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

from cls_module.cls import CLS

import utils

from datasets.tfms import NoiseTransformation, OcclusionTransformation
from datasets.omniglot_one_shot_dataset import OmniglotTransformation, OmniglotOneShotDataset
from datasets.omniglot_instance_dataset import OmniglotInstanceDataset
from datasets.cifar_one_shot_dataset import CifarTransformation, CifarOneShotDataset
from oneshot_metrics import OneshotMetrics

LOG_EVERY = 20
LOG_EVERY_EVAL = 1
VAL_EVERY = 20
VALIDATE = True
SAVE_EVERY = 1
MAX_VAL_STEPS = 100
MAX_PRETRAIN_STEPS = -1
VAL_SPLIT = 0.175
SAVE_RUN_MODEL = False
LOAD_LTM_ONLY = False

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

  seed = config['seed']
  utils.set_seed(seed)
  np.random.seed(seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


  dataset_name = config.get('dataset')

  if dataset_name == 'Omniglot' or dataset_name == 'Omniglot-Instance':
    image_tfms = transforms.Compose([
        transforms.ToTensor(),
        OmniglotTransformation(resize_factor=config['image_resize_factor'])
    ])

  elif dataset_name == 'CIFAR':
    image_tfms = transforms.Compose([
      transforms.ToTensor(),
      CifarTransformation(resize_factor=config['image_resize_factor'])
    ])

  image_shape = config['image_shape']
  pretrained_model_path = config.get('pretrained_model_path', None)
  previous_run_path = config.get('previous_run_path', None)
  train_from = config.get('train_from', None)

  # Pretraining
  # ---------------------------------------------------------------------------
  start_epoch = 1

  experiment = config.get('experiment_name')
  now = datetime.datetime.now()
  experiment_time = now.strftime("%Y%m%d-%H%M%S")
  if previous_run_path:
    summary_dir = previous_run_path
    writer = SummaryWriter(log_dir=summary_dir)
    model = CLS(image_shape, config, device=device, writer=writer).to(device)

    # Ensure that pretrained model path doesn't exist so that training occurs
    #pretrained_model_path = None

    if train_from == 'scratch':
      # Clear the directory
      for file in os.listdir(previous_run_path):
        path = os.path.join(previous_run_path, file)
        try:
          os.unlink(os.path.join(path))
        except Exception as e:
          print(f"Failed to remove file with path {path} due to exception {e}")

    elif train_from == 'latest':
      model_path = os.path.join(previous_run_path, 'pretrained_model_*')
      # Find the latest file in the directory
      latest = max(glob.glob(model_path), key=os.path.getctime)

      # Find the epoch that was stopped on
      i = len(latest) - 4  # (from before the .pt)
      latest_epoch = 0
      while latest[i] != '_':
        latest_epoch *= 10
        latest_epoch += int(latest[i])
        i -= 1

      if latest_epoch < config['pretrain_epochs']:
        start_epoch = latest_epoch + 1

      print("Attempting to find existing checkpoint")
      if os.path.exists(latest):
        try:
          model.load_state_dict(torch.load(latest))
        except Exception as e:
          print("Failed to load model from path: {latest}. Please check path and try again due to exception {e}.")
          return

  else:
    summary_dir = utils.get_summary_dir(experiment, experiment_time, seed)
    writer = SummaryWriter(log_dir=summary_dir)
    model = CLS(image_shape, config, device=device, writer=writer).to(device)

  if not pretrained_model_path:

    if dataset_name == 'Omniglot' or dataset_name == 'Omniglot-Instance':
      dataset = datasets.Omniglot('./data', background=True, download=True, transform=image_tfms)

    elif dataset_name == 'CIFAR':
      dataset = datasets.CIFAR10('./data', download=True, transform=image_tfms)


    val_size = round(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['pretrain_batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['pretrain_batch_size'], shuffle=True)

    # Pre-train the model
    for epoch in range(start_epoch, config['pretrain_epochs'] + 1):

      for batch_idx, (data, target) in enumerate(train_loader):

        if 0 < MAX_PRETRAIN_STEPS < batch_idx:
          print("Pretrain steps, {}, has exceeded max of {}.".format(batch_idx, MAX_PRETRAIN_STEPS))
          break

        data, target = data.to(device), target.to(device)
        losses, _ = model(data, labels=target if model.is_ltm_supervised() else None, mode='pretrain')
        pretrain_loss = losses['ltm']['memory']['loss'].item()

        if batch_idx % LOG_EVERY == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx, len(train_loader),
            100. * batch_idx / len(train_loader), pretrain_loss))

        if VALIDATE and ((batch_idx + 1) % VAL_EVERY == 0 or (batch_idx + 1) == len(train_loader)):
          logging.info("\t--- Start validation")


          with torch.no_grad():
            for batch_idx_val, (val_data, val_target) in enumerate(val_loader):

              if batch_idx_val >= MAX_VAL_STEPS:
                print("\tval batch steps, {}, has exceeded max of {}.".format(batch_idx_val, MAX_VAL_STEPS))
                break

              val_data, val_target = val_data.to(device), val_target.to(device)

              val_losses, _ = model(val_data, labels=val_target if model.is_ltm_supervised() else None, mode='validate')
              val_pretrain_loss = val_losses['ltm']['memory']['loss'].item()

              if batch_idx_val % LOG_EVERY == 0:
                print('\tValidation for Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx_val, len(val_loader),
                  100. * batch_idx_val / len(val_loader), val_pretrain_loss))

      if epoch % SAVE_EVERY == 0:
        pretrained_model_path = os.path.join(summary_dir, 'pretrained_model_' + str(epoch) + '.pt')
        print('Saving model to:', pretrained_model_path)
        torch.save(model.state_dict(), pretrained_model_path)

  # Study and Recall
  # ---------------------------------------------------------------------------
  print('-------- Few-shot Evaluation (Study and Recall) ---------')

  # Prepare data transformations
  recall_tfms_opts =[
      transforms.ToTensor(),
      OmniglotTransformation(resize_factor=config['image_resize_factor']),
  ]

  if config.get('noise_type', None):
    recall_tfms_opts.append(
        NoiseTransformation(noise_type=config.get('noise_type'),
                            noise_factor=config.get('noise_factor'))
    )

  if config.get('degrade_type', None):
    recall_tfms_opts.append(
        OcclusionTransformation(degrade_type=config.get('degrade_type'),
                                degrade_factor=config.get('degrade_factor'))
    )

  study_tfms = image_tfms
  recall_tfms = transforms.Compose(recall_tfms_opts)

  # Prepare data loaders
  if dataset_name == 'Omniglot':
    study_dataset = OmniglotOneShotDataset('./data', train=True, download=True,
                               transform=study_tfms, target_transform=None)
    recall_dataset = OmniglotOneShotDataset('./data', train=False, download=True,
                               transform=recall_tfms, target_transform=None)

  if dataset_name == 'Omniglot-Instance':
    # Create the study dataset and generate show/match sets of instances in the same class
    study_dataset = OmniglotInstanceDataset('./data', train=True, download=True,
                               transform=study_tfms, target_transform=None,
                               batch_size=config['study_batch_size'])

    # Pass the generated show/match sets to ensure that the study and recall modes
    # are consistently paired and avoid re-generating the sets
    recall_dataset = OmniglotInstanceDataset('./data', train=False, download=True,
                               transform=recall_tfms, target_transform=None,
                               batch_size=config['study_batch_size'],
                               show_filenames=study_dataset.show_filenames,
                               show_labels=study_dataset.show_labels,
                               match_labels=study_dataset.match_labels,
                               match_filenames=study_dataset.match_filenames)

  elif dataset_name == 'CIFAR':
    rand_class = np.random.randint(low=0, high=64, size=20)  # array of 20 random integers to select classes
    study_dataset = CifarOneShotDataset('./data', mode='train', transform=study_tfms, target_transform=None,
                        classes=rand_class, download=False)
    recall_dataset = CifarOneShotDataset('./data', mode='test', transform=recall_tfms, target_transform=None,
                        classes=rand_class, download=False)

  study_loader = torch.utils.data.DataLoader(study_dataset, batch_size=config['study_batch_size'], shuffle=False)
  recall_loader = torch.utils.data.DataLoader(recall_dataset, batch_size=config['study_batch_size'], shuffle=False)

  assert len(study_loader) == len(recall_loader)

  oneshot_dataset = enumerate(zip(study_loader, recall_loader))

  # Initialise metrics
  oneshot_metrics = OneshotMetrics()

  # Initialise CLS
  model = CLS(image_shape, config, device=device, writer=writer).to(device)

  for idx, ((study_data, study_target), (recall_data, recall_target)) in oneshot_dataset:
    if LOG_EVERY_EVAL > 0 and idx % LOG_EVERY_EVAL == 0:
      print('Step #{}'.format(idx))

    study_data = study_data.to(device)
    study_target = torch.from_numpy(np.array(study_target)).to(device)
    recall_data = recall_data.to(device)
    recall_target = torch.from_numpy(np.array(recall_target)).to(device)

    # Reset to saved model
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    if LOAD_LTM_ONLY:
      pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('ltm')}

    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)

    model.reset()

    # Study
    # --------------------------------------------------------------------------
    for step in range(config['study_steps']):
      study_train_losses, _ = model(study_data, study_target, mode='study')
      study_train_loss = study_train_losses['stm']['memory']['loss']

      # if all (k in study_train_loss for k in ('pr', 'pm_ec')):
      #   print('Losses step {}, ite {}: \t PR:{:.6f}\
      #         PR mismatch: {:.6f} \t PM-EC: {:.6f}'.format(idx, step,
      #                                                         study_train_loss['pr'].item(),
      #                                                         study_train_loss['pr_mismatch'].item(),
      #                                                         study_train_loss['pm_ec'].item()))

      model(study_data, study_target, mode='recall')

    # Recall
    # --------------------------------------------------------------------------
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

      # PR Accuracy (study first) - this is the version used in the paper
      if 'stm_pr' in model.features['study']:
        oneshot_metrics.compare(prefix='pr_sf_',
                                primary_features=model.features['study']['stm_pr'],
                                primary_labels=model.features['study']['labels'],
                                secondary_features=model.features['recall']['stm_pr'],
                                secondary_labels=model.features['recall']['labels'],
                                comparison_type='match_mse')

      oneshot_metrics.report()

      summary_names = [
        'study_inputs',
        'study_stm_pr',
        'study_stm_ca3',

        'recall_inputs',
        'recall_stm_pr',
        'recall_stm_ca3',
        'recall_stm_recon'
      ]

      summary_images = []
      for name in summary_names:
        mode_key, feature_key = name.split('_', 1)

        if not feature_key in model.features[mode_key]:
          continue

        summary_features = model.features[mode_key][feature_key]

        if len(summary_features.shape) > 2:
          summary_features = summary_features.permute(0, 2, 3, 1)

        summary_shape, _ = utils.square_image_shape_from_1d(np.prod(summary_features.data.shape[1:]))
        summary_shape[0] = summary_features.data.shape[0]

        summary_image = (name, summary_features, summary_shape)
        summary_images.append(summary_image)

      utils.add_completion_summary(summary_images, summary_dir, idx, save_figs=True)

    # Optional: Save the model checkpoint for this run
    if SAVE_RUN_MODEL:
      run_model_path = os.path.join(summary_dir, 'run_model_' + str(idx) + '.pt')
      print('Saving model to:', run_model_path)
      torch.save(model.state_dict(), run_model_path)

  oneshot_metrics.report_averages()

  writer.flush()
  writer.close()


if __name__ == '__main__':
  main()
