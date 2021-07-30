import os
import json
import argparse
import logging
import utils
import torch
import numpy as np
import glob
from torch.utils.tensorboard import SummaryWriter
from cls_module.cls import CLS
from sequence_generator import SequenceGenerator
from omniglot_one_shot_dataset import OmniglotTransformation
from omniglot_per_alphabet_dataset import OmniglotAlphabet
from torchvision import transforms
from oneshot_metrics import OneshotMetrics

LOG_EVERY = 20
LOG_EVERY_EVAL = 1
VAL_EVERY = 20
SAVE_EVERY = 1
MAX_VAL_STEPS = 100
MAX_PRETRAIN_STEPS = -1
VAL_SPLIT = 0.175

def main():
    parser = argparse.ArgumentParser(description='Pair structure. Replicate of Schapiro')
    parser.add_argument('-c', '--config', nargs="?", type=str, default='./definitions/aha_config_pairs.json',
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

    image_tfms = transforms.Compose([
       transforms.ToTensor(),
       OmniglotTransformation(resize_factor=config['image_resize_factor'])])

    image_shape = config['image_shape']
    alphabet_name = config.get('alphabet')

    pretrained_model_path = config.get('pretrained_model_path', None)
    previous_run_path = config.get('previous_run_path', None)
    train_from = config.get('train_from', None)

    # Pretraining
    # ---------------------------------------------------------------------------
    start_epoch = 1

    experiment = config.get('experiment_name')

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
            print(model_path)
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
        summary_dir = utils.get_summary_dir(experiment)
        writer = SummaryWriter(log_dir=summary_dir)
        model = CLS(image_shape, config, device=device, writer=writer).to(device)

    if not pretrained_model_path:
        dataset = OmniglotAlphabet('./data', alphabet_name, False, writer_idx='any', download=True, transform=image_tfms,
                                   target_transform=None)

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

                # An empty image is add to each character (an empty pair character)
                data = [torch.cat((a, torch.zeros(1, 52, 52)), 2) for a in data]
                data = torch.stack(data)
                labels = list(set(target))
                target = torch.tensor([labels.index(value) for value in target])
                data = data.to(device)
                target = target.to(device)

                losses, _ = model(data, labels=target if model.is_ltm_supervised() else None, mode='pretrain')
                pretrain_loss = losses['ltm']['memory']['loss'].item()

                if batch_idx % LOG_EVERY == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx, len(train_loader),
                        100. * batch_idx / len(train_loader), pretrain_loss))

                if batch_idx % VAL_EVERY == 0 or batch_idx == len(train_loader) - 1:
                    logging.info("\t--- Start validation")

                    with torch.no_grad():
                        for batch_idx_val, (val_data, val_target) in enumerate(val_loader):

                            if batch_idx_val >= MAX_VAL_STEPS:
                                print("\tval batch steps, {}, has exceeded max of {}.".format(batch_idx_val,
                                                                                              MAX_VAL_STEPS))
                                break

                            val_data = [torch.cat((a, torch.zeros(1, 52, 52)), 2) for a in val_data]
                            val_data = torch.stack(val_data)

                            val_data = val_data.to(device)
                            target = target.to(device)

                            val_losses, _ = model(val_data, labels=val_target if model.is_ltm_supervised() else None,
                                                  mode='validate')
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

    characters = config.get('characters')
    length = config.get('sequence_length')

    # Create sequence of pairs
    sequence_study = SequenceGenerator(characters, length)
    sequence_recall = SequenceGenerator(characters, length)

    sequence_study_tensor = torch.FloatTensor(sequence_study.sequence)
    sequence_recall_tensor = torch.FloatTensor(sequence_recall.sequence)
    study_loader = torch.utils.data.DataLoader(sequence_study_tensor, batch_size=config['study_batch_size'], shuffle=False)
    recall_loader = torch.utils.data.DataLoader(sequence_recall_tensor, batch_size=config['study_batch_size'], shuffle=False)

    pair_sequence_dataset = enumerate(zip(study_loader, recall_loader))

    # Load images from the selected alphabet from a specific writer or random writers
    variation = config.get('variation')
    idx_study = config.get('character_idx_study')
    idx_recall = config.get('character_idx_recall')

    alphabet = OmniglotAlphabet('./data', alphabet_name, True, False, idx_study, download=True, transform=image_tfms, target_transform=None)
    alphabet_recall = OmniglotAlphabet('./data', alphabet_name, True, variation, idx_recall, download=True, transform=image_tfms,
                                target_transform=None)


    # Initialise metrics
    oneshot_metrics = OneshotMetrics()

    # Load the pretrained model
    model = CLS(image_shape, config, device=device, writer=writer).to(device)

    for idx, (study_set, recall_set) in pair_sequence_dataset:
        study_data = [torch.cat((alphabet[int(a[0].item())][0], alphabet[int(a[1].item())][0]), 2) for a in study_set]
        study_target_pairs = [(alphabet[int(a[0].item())][1], alphabet[int(a[1].item())][1]) for a in study_set]
        labels_study = list(set(study_target_pairs))
        study_target = [labels_study.index(value) for value in study_target_pairs]

        study_data = torch.stack(study_data)
        study_target = torch.tensor(study_target, dtype=torch.long, device=device)

        study_data = study_data.to(device)
        study_target = study_target.to(device)

        recall_data = [torch.cat((alphabet_recall[int(a[0].item())][0], alphabet_recall[int(a[1].item())][0]), 2)
                       for a in recall_set]
        recall_target_pairs = [(alphabet_recall[int(a[0].item())][1], alphabet_recall[int(a[1].item())][1]) for a in
                               recall_set]

        del recall_data[0:characters]

        recall_data = torch.stack(recall_data)
        single_characters = [torch.cat((alphabet_recall[a][0], torch.zeros(1, 52, 52)), 2) for a in range(0, characters)]
        single_characters = torch.stack(single_characters)
        recall_data = torch.cat((single_characters, recall_data), 0)

        labels_recall = list(set(recall_target_pairs))
        recall_target = [labels_recall.index(value) for value in recall_target_pairs]
        del recall_target[0:characters-1]

        recall_target = list(range(max(recall_target) + 1, max(recall_target) + characters)) + recall_target
        recall_target = torch.tensor(recall_target, dtype=torch.long, device=device)

        recall_data = recall_data.to(device)
        recall_target = recall_target.to(device)

        assert len(study_data) == len(recall_data)

        # Reset to saved model
        model.load_state_dict(torch.load(pretrained_model_path))
        model.reset()

        # Study
        # --------------------------------------------------------------------------
        for step in range(config['study_steps']):
            study_train_losses, _ = model(study_data, study_target, mode='study')
            study_train_loss = study_train_losses['stm']['memory']['loss']

            print('Losses step {}, ite {}: \t PR:{:.6f}\
                PR mismatch: {:.6f} \t PM-EC: {:.6f}'.format(idx, step,
                                                             study_train_loss['pr'].item(),
                                                             study_train_loss['pr_mismatch'].item(),
                                                             study_train_loss['pm_ec'].item()))

            model(recall_data, recall_target, mode='study_validate')

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
                    secondary_feature = utils.find_json_value(metrics_config['secondary_feature_names'][i],
                                                              model.features)
                    secondary_label = utils.find_json_value(metrics_config['secondary_label_names'][i], model.features)
                    comparison_type = metrics_config['comparison_types'][i]
                    prefix = metrics_config['prefixes'][i]

                    oneshot_metrics.compare(prefix,
                                            primary_feature, primary_label,
                                            secondary_feature, secondary_label,
                                            comparison_type=comparison_type)

            # PR Accuracy (study first) - this is the version used in the paper
            oneshot_metrics.compare(prefix='pr_sf_',
                                    primary_features=model.features['study']['stm_pr'],
                                    primary_labels=model.features['study']['labels'],
                                    secondary_features=model.features['recall']['stm_pr'],
                                    secondary_labels=model.features['recall']['labels'],
                                    comparison_type='match_mse')

            # oneshot_metrics.compare(prefix='pr_rf_',
            #                         primary_features=model.features['recall']['stm_pr'],
            #                         primary_labels=model.features['recall']['labels'],
            #                         secondary_features=model.features['study']['stm_pr'],
            #                         secondary_labels=model.features['study']['labels'],
            #                         comparison_type='match_mse')

            oneshot_metrics.report()

            summary_names = [
                'study_inputs',
                'study_stm_pr',
                'study_stm_pc',

                'recall_inputs',
                'recall_stm_pr',
                'recall_stm_pc',
                'recall_stm_recon'
            ]

            summary_images = []
            for name in summary_names:
                mode_key, feature_key = name.split('_', 1)

                summary_features = model.features[mode_key][feature_key]
                if len(summary_features.shape) > 2:
                    summary_features = summary_features.permute(0, 2, 3, 1)

                summary_shape, _ = utils.square_image_shape_from_1d(np.prod(summary_features.data.shape[1:]))
                summary_shape[0] = summary_features.data.shape[0]

                summary_image = (name, summary_features, summary_shape)
                summary_images.append(summary_image)

            utils.add_completion_summary(summary_images, summary_dir, idx, save_figs=True)

    oneshot_metrics.report_averages()

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
