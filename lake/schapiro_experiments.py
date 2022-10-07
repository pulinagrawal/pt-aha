import os
import csv
import json
import argparse
import logging
import utils
import datetime
import glob

import torch
import torch.nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from scipy import stats

from cls_module.cls import CLS

from datasets.sequence_generator import SequenceGenerator, SequenceGeneratorGraph, SequenceGeneratorTriads
from datasets.omniglot_one_shot_dataset import OmniglotTransformation
from datasets.omniglot_per_alphabet_dataset import OmniglotAlphabet
from embeddings import Correlations, Overlap
from Visualisations import HeatmapPlotter, BarPlotter
from Visualisations import FrequencyPlotter
from oneshot_metrics import OneshotMetrics

LOG_EVERY = 20
LOG_EVERY_EVAL = 1
VAL_EVERY = 2
SAVE_EVERY = 1
MAX_VAL_STEPS = 100
MAX_PRETRAIN_STEPS = -1
VAL_SPLIT = 0.175

def main():
    parser = argparse.ArgumentParser(description='Pair structure. Replicate of Schapiro')
    parser.add_argument('-c', '--config', nargs="?", type=str, help='Configuration file for experiments.')
    parser.add_argument('-l', '--logging', nargs="?", type=str, default='warning',
                        help='Logging level.')
    args = parser.parse_args()

    logging_level = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=logging_level)

    with open(args.config) as config_file:
        print('loading config', config_file)
        config = json.load(config_file)

    seed_ltm = config['seeds']
    seeds = config['seeds']
    experiment = config.get('experiment_name')
    learning_type = config.get('learning_type')
    components = config.get('test_components')
    now = datetime.datetime.now()
    experiment_time = now.strftime("%Y%m%d-%H%M%S")

    if config['own_alphabet']:
        variation_recall = False
        variation_training = False
        writer_idx_study = 0
        writer_idx_validation = 0
        writer_idx_recall = 0
    else:
        variation_recall = config['variation_recall']
        variation_training = config['variation_training']
        writer_idx_study = config['writer_idx_study']
        writer_idx_validation = config['writer_idx_validation']
        writer_idx_recall = config['writer_idx_recall']

    if experiment not in ["pairs_structure", "community_structure", "associative_inference"]:
        print("Experiment NOT specified. Must be pairs_structure, community_structure or associative_inference ")
        exit()

    main_summary_dir = utils.get_summary_dir(experiment + "/" + learning_type, experiment_time,
                                             main_folder=True)

    with open(main_summary_dir + '/info_exp.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_tfms = transforms.Compose([
        transforms.ToTensor(),
        OmniglotTransformation(resize_factor=config['image_resize_factor'])])

    alphabet_name = config.get('alphabet')
    pretrained_model_path = config.get('pretrained_model_path', None)
    pearson_r_test = torch.zeros((1, 2))
    if experiment == 'community_structure':
        pearson_r_test = torch.zeros((1, 4))
    pearson_r_test = {k: pearson_r_test for k in components}
    characters = config.get('characters')
    pearson_r_tensor = torch.zeros((characters, characters))
    pearson_r_tensor = pearson_r_tensor[None, :, :]
    pearson_r_tensor = {k: pearson_r_tensor for k in components}

    for _ in range(seeds):
        seed = np.random.randint(1, 10000)
        utils.set_seed(seed)

        # Pretraining
        # ---------------------------------------------------------------------------
        summary_dir = utils.get_summary_dir(experiment + "/" + learning_type, experiment_time, main_folder=False, seed=str(seed))
        writer = SummaryWriter(log_dir=summary_dir)
        if pretrained_model_path:
            # summary_dir = pretrained_model_path
             #writer = SummaryWriter(log_dir=summary_dir)
            model = CLS(config['image_shape'], config, device=device, writer=writer, output_shape=config['pairs_shape']).to(device)

            model_path = os.path.join(pretrained_model_path, 'pretrained_model_*')
            print(model_path)
            # Find the latest file in the directory
            latest = max(glob.glob(model_path), key=os.path.getctime)

            trained_model_path = latest

            # Load only the LTM part
            pretrained_dict = torch.load(trained_model_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('ltm')}
            model.load_state_dict(pretrained_dict)

        else:
            start_epoch = 1
            utils.set_seed(seed_ltm)
            model = CLS(config['image_shape'], config, device=device, writer=writer, output_shape=config['pairs_shape']).to(device)

            dataset = OmniglotAlphabet('./data', alphabet_name, False, own_alphabet=config['own_alphabet'],
                                       download=True,
                                       transform=image_tfms,
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

                    data = data.to(device)
                    target = target.to(device)

                    losses, _ = model(data, labels=target if model.is_ltm_supervised() else None, mode='pretrain')
                    pretrain_loss = losses['ltm']['memory']['loss'].item()

                    if batch_idx % LOG_EVERY == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx, len(train_loader),
                            100. * batch_idx / len(train_loader), pretrain_loss))

                if (epoch + 1) % VAL_EVERY == 0:
                    logging.info("\t--- Start validation")

                    with torch.no_grad():
                        for batch_idx_val, (val_data, val_target) in enumerate(val_loader):

                            if batch_idx_val >= MAX_VAL_STEPS:
                                print("\tval batch steps, {}, has exceeded max of {}.".format(batch_idx_val,
                                                                                              MAX_VAL_STEPS))
                                break


                            val_data = val_data.to(device)
                            val_target = val_target.to(device)

                            val_losses, _ = model(val_data, labels=val_target if model.is_ltm_supervised() else None,
                                                  mode='validate_ltm')
                            val_pretrain_loss = val_losses['ltm']['memory']['loss'].item()

                            if batch_idx_val % LOG_EVERY == 0:
                                print('\tValidation for Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, batch_idx_val, len(val_loader),
                                    100. * batch_idx_val / len(val_loader), val_pretrain_loss))

                if epoch % SAVE_EVERY == 0:
                    trained_model_path = os.path.join(summary_dir, 'pretrained_model_' + str(epoch) + '.pt')
                    print('Saving model to:', trained_model_path)
                    torch.save(model.state_dict(), trained_model_path)
                pretrained_model_path = summary_dir

        # Study and Recall
        # ---------------------------------------------------------------------------
        print('-------- Few-shot Evaluation (Study and Recall) ---------')

        length = config['sequence_length']
        batch_size = config['study_batch_size']


        # Create sequence
        if experiment == "pairs_structure":
            sequence_study = SequenceGenerator(characters, length, learning_type)
            sequence_validation = SequenceGenerator(characters, length, learning_type)
        elif experiment == "community_structure":
            sequence_study = SequenceGeneratorGraph(characters, length, learning_type, config['communities'])
            sequence_validation = SequenceGeneratorGraph(characters, length, learning_type, config['communities'])
        elif experiment == "associative_inference":
            sequence_study = SequenceGeneratorTriads(characters, length, learning_type, batch_size)
            sequence_validation = SequenceGeneratorTriads(characters, length, learning_type, batch_size)

        predictions = []
        pairs_inputs = []

        sequence_study_tensor = torch.FloatTensor(sequence_study.sequence)
        sequence_validation_tensor = torch.FloatTensor(sequence_validation.sequence)
        study_loader = torch.utils.data.DataLoader(sequence_study_tensor, batch_size=batch_size, shuffle=False)
        validation_loader = torch.utils.data.DataLoader(sequence_validation_tensor, batch_size=batch_size, shuffle=False)


        # Load images from the selected alphabet from a specific writer or random writers
        alphabet = OmniglotAlphabet('./data', alphabet_name, True,
                                    own_alphabet=config['own_alphabet'],
                                    variation=variation_training,
                                    writer_idx=writer_idx_study, download=True,
                                    transform=image_tfms, target_transform=None)

        alphabet_validation = OmniglotAlphabet('./data', alphabet_name, True,
                                               own_alphabet=config['own_alphabet'],
                                               variation=variation_training,
                                               writer_idx=writer_idx_validation, download=True,
                                               transform=image_tfms, target_transform=None)

        alphabet_recall = OmniglotAlphabet('./data', alphabet_name, True,
                                           own_alphabet=config['own_alphabet'],
                                           variation=variation_recall,
                                           writer_idx=writer_idx_recall, download=True,
                                           transform=image_tfms, target_transform=None)

        labels_study = sequence_study.core_label_sequence

        main_pairs, _, main_pairs_paired = format_data(alphabet, labels_study, labels_study, model, device,
                                              config['activation_coefficient'],
                                              config['overlap_option'])
        main_pairs_paired_flat = torch.flatten(main_pairs_paired, start_dim=1)


        single_characters = [alphabet_recall[a][0] for a in range(0, characters)]
        single_characters = torch.stack(single_characters)


        # Recall data size must be the batch size
        recall_data = single_characters.repeat(-(-batch_size // characters), 1, 1, 1)
        recall_data = recall_data[0:batch_size]
        recall_paired_data = torch.cat([recall_data, torch.zeros_like(recall_data)], dim=3)

        recall_target = list(range(single_characters.size()[0]))
        recall_target = recall_target * -(-batch_size // characters)
        recall_target = recall_target[0:batch_size]
        recall_target = torch.tensor(recall_target, dtype=torch.long, device=device)

        recall_data = recall_data.to(device)
        recall_target = recall_target.to(device)

        _, recall_data = model(recall_data, recall_target, mode='extractor')
        recall_data = recall_data['ltm']['memory']['output'].detach().cpu()

        correlator = Correlations(main_summary_dir, seed)
        correlator.correlation(recall_data[0:characters], recall_data[0:characters], 'single_characters')
        correlator.correlation(recall_data[0:characters], main_pairs, 'main_pairs_vs_characters')
        correlator.correlation(main_pairs, main_pairs, 'main_pairs')

        # Initialise metrics
        oneshot_metrics = OneshotMetrics()

        for stm_epoch in range(config['train_epochs']):

            pair_sequence_dataset = enumerate(zip(study_loader, validation_loader))


            for idx, (study_set, validation_set) in pair_sequence_dataset:

                # Reset
                model.reset()

                if learning_type == 'statistical':
                    frequency = FrequencyPlotter(main_summary_dir, study_set, str(seed) + '_' + str(idx))
                    frequency.create_bar()

                study_data, study_target, study_paired_data = format_data(alphabet, study_set, labels_study, model, device,
                                                                          config['activation_coefficient'],
                                                                          config['overlap_option'])

                study_data = study_data.to(device)
                study_target = study_target.to(device)
                study_paired_data = study_paired_data.to(device)

                val_data, val_target, val_paired_data = format_data(alphabet_validation, validation_set, labels_study, model, device,
                                                                          config['activation_coefficient'],
                                                                          config['overlap_option'])
                val_data = val_data.to(device)
                val_target = val_target.to(device)
                val_paired_data = val_paired_data.to(device)

                # Study
                # --------------------------------------------------------------------------
                for step in range(config['late_response_steps']):

                        study_train_losses, _ = model(study_data, study_target, mode='study', ec_inputs=study_data,
                                                          paired_inputs=study_paired_data)
                        study_train_loss = study_train_losses['stm']['memory']['loss']

                        if config.get('stm').get('hebbian_perforant'):
                            print('Losses batch {}, ite {}: \t EC_CA3:{:.6f}\
                             \t ca3_ca1: {:.6f}'.format(idx, step, study_train_loss['ec_ca3'].item(),
                                                        study_train_loss['ca3_ca1'].item()))
                        else:
                            print('Losses batch {}, ite {}: \t PR:{:.6f}\
                            PR mismatch: {:.6f} \t ca3_ca1: {:.6f}'.format(idx, step,
                                                                           study_train_loss['pr'].item(),
                                                                           study_train_loss['pr_mismatch'].item(),
                                                                           study_train_loss['ca3_ca1'].item()))

                        validation_losses, _ = model(val_data, val_target, mode='validate', ec_inputs=val_data,
                                                     paired_inputs=val_paired_data)

                        if step == (config['early_response_step']-1) or step == (config['late_response_steps']-1):
                            with torch.no_grad():
                                recall_data = recall_data.to(device)
                                recall_target = recall_target.to(device)
                                recall_paired_data = recall_paired_data.to(device)

                                _, recall_outputs = model(recall_data, recall_target, mode='recall', ec_inputs=recall_data,
                                                          paired_inputs=recall_paired_data)

                                # Perform another iteration using recalled outputs
                                if config.get('recurrence_steps', 0) > 0:
                                  for rstep in range(config['recurrence_steps']):
                                    bigloop_ec_inputs = recall_outputs["stm"]["memory"]['ca1']['decoding']
                                    bigloop_paired_inputs = recall_outputs["stm"]["memory"]['decoding']
                                    _, recall_outputs = model(bigloop_ec_inputs, recall_target, mode='recall', ec_inputs=bigloop_ec_inputs,
                                                              paired_inputs=bigloop_paired_inputs)

                                cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                                for component in components:
                                    if component == 'dg':
                                        recall_outputs_flat = torch.flatten(recall_outputs["stm"]["memory"][component],
                                                                        start_dim=1)
                                    if component == 'pr':
                                        recall_outputs_flat = torch.flatten(recall_outputs["stm"]["memory"][component]['pr_out'],
                                                                            start_dim=1)
                                    if component == 'ec_ca3':
                                        recall_outputs_flat = torch.flatten(recall_outputs["stm"]["memory"][component]['ca3_cue'],
                                                start_dim=1)
                                    if component == 'ca3':
                                        recall_outputs_flat = torch.flatten(recall_outputs["stm"]["memory"][component],
                                                                        start_dim=1)
                                    if component == 'ca3_ca1':
                                        recall_outputs_flat = torch.flatten(
                                            recall_outputs["stm"]["memory"][component]['decoding'],
                                            start_dim=1)
                                    if component == 'ca1_enc':
                                        recall_outputs_flat = torch.flatten(
                                          recall_outputs["stm"]["memory"]['ca1']['encoding'],
                                          start_dim=1)
                                    if component == 'ca1_dec':
                                        recall_outputs_flat = torch.flatten(
                                            recall_outputs["stm"]["memory"]['ca1']['decoding'],
                                            start_dim=1)
                                        if experiment == "associative_inference" and step == (config['late_response_steps']-1):
                                            correlator.transitivity(recall_outputs_flat[0:characters], recall_data[0:characters])
                                    if component == 'recon_pair':
                                        recall_outputs_flat = torch.flatten(
                                            recall_outputs["stm"]["memory"]['decoding'],
                                            start_dim=1)
                                        similarity = [[cos(a, b) for b in main_pairs_paired_flat.cpu()] for a in recall_outputs_flat.cpu()]
                                        similarity_idx = [t.index(max(t)) for t in similarity]
                                        predictions.extend([[labels_study[a] for a in similarity_idx]])

                                    recall_outputs_flat = recall_outputs_flat[0:characters].cpu()
                                    pearson_r = [[stats.pearsonr(a, b)[0] for a in recall_outputs_flat] for b in
                                                 recall_outputs_flat]
                                    pearson_r = torch.tensor(pearson_r, dtype=torch.float32)
                                    pearson_r_tensor[component] = torch.cat((pearson_r_tensor[component], pearson_r[None, :, :]), 0)

                                    if experiment == 'pairs_structure':
                                        pearson_all_pattern = sum([pearson_r[i, j] for (i, j) in sequence_study.test_sequence])/len(sequence_study.test_sequence)
                                        pearson_core_pattern = sum([pearson_r[i, j] for (i, j) in sequence_study.core_sequence]) / len(sequence_study.core_sequence)
                                        tmp_pearson_test = torch.tensor([[pearson_core_pattern, pearson_all_pattern]])
                                    if experiment == 'associative_inference':
                                        pearson_transitive_pattern = sum([pearson_r[i, j] for (i, j) in sequence_study.test_sequence]) / len(sequence_study.test_sequence)
                                        pearson_direct_pattern = sum([pearson_r[i, j] for (i, j) in sequence_study.core_sequence]) / len(sequence_study.core_sequence)
                                        pearson_base_pattern = sum([pearson_r[i, j] for (i, j) in sequence_study.base_sequence]) / len(sequence_study.base_sequence)
                                        tmp_pearson_test = torch.tensor([[pearson_transitive_pattern - pearson_base_pattern, pearson_direct_pattern-pearson_base_pattern]])
                                    if experiment == 'community_structure':
                                        pearson_within_internal = sum([pearson_r[i, j] for (i, j) in sequence_study.graph_sequences[0]]) / len(sequence_study.graph_sequences[0])
                                        pearson_within_boundary = sum([pearson_r[i, j] for (i, j) in sequence_study.graph_sequences[1]]) / len(sequence_study.graph_sequences[1])
                                        pearson_across_boundary = sum([pearson_r[i, j] for (i, j) in sequence_study.graph_sequences[2]]) / len(sequence_study.graph_sequences[2])
                                        pearson_across_other = sum([pearson_r[i, j] for (i, j) in sequence_study.graph_sequences[3]]) / len(sequence_study.graph_sequences[3])
                                        tmp_pearson_test = torch.tensor([[pearson_within_internal, pearson_within_boundary, pearson_across_boundary, pearson_across_other]])

                                    pearson_r_test[component] = torch.cat((pearson_r_test[component], tmp_pearson_test), 0)

                pairs_inputs.extend([[(int(a[0]), int(a[1])) for a in study_set]])


                # --------------------------------------------------------------------------
                with torch.no_grad():

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

                        if config.get('stm').get('hebbian_perforant'):
                            stm_feature = 'stm_ca3_cue'
                        else:
                            stm_feature = 'stm_pr'

                        oneshot_metrics.compare(prefix='pr_rf_',
                                                 primary_features=model.features['recall'][stm_feature],
                                                 primary_labels=model.features['recall']['labels'],
                                                 secondary_features=model.features['study'][stm_feature],
                                                 secondary_labels=model.features['study']['labels'],
                                                 comparison_type='match_mse')

                        oneshot_metrics.report()

                        summary_names = [
                            'study_paired_inputs',
                            'study_' + stm_feature,
                            'study_stm_ca3',

                            'recall_paired_inputs',
                            'recall_' + stm_feature,
                            'recall_stm_ca3',
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

                        utils.add_completion_summary(summary_images, summary_dir, str(idx) + '_' + str(stm_epoch) + '_'
                                                     + str(seed), save_figs=True)

        # Save results
        predictions_early = predictions[0:predictions.__len__():2]
        predictions_late = predictions[1:predictions.__len__():2]

        with open(main_summary_dir + '/predictions_early_' + str(seed) + '.csv', 'w', encoding='UTF8') as f:
            writer_file = csv.writer(f)
            writer_file.writerows(predictions_early)

        with open(main_summary_dir + '/predictions_late_' + '_' + str(seed)+'.csv', 'w', encoding='UTF8') as f:
            writer_file = csv.writer(f)
            writer_file.writerows(predictions_late)

        with open(main_summary_dir + '/pair_inputs_'+ str(seed) + '.csv', 'w', encoding='UTF8') as f:
            writer_file = csv.writer(f)
            writer_file.writerows(pairs_inputs)

        oneshot_metrics.report_averages()
        writer.flush()
        writer.close()

    pearson_r_early = {a: pearson_r_tensor[a][1:pearson_r_tensor[a].shape[0]:2] for a in pearson_r_tensor}
    pearson_r_late = {a: pearson_r_tensor[a][2:pearson_r_tensor[a].shape[0] + 1:2] for a in pearson_r_tensor}
    pearson_r_early = {a: torch.mean(pearson_r_early[a], 0) for a in pearson_r_early}
    pearson_r_late = {a: torch.mean(pearson_r_late[a], 0) for a in pearson_r_late}

    pearson_r_early_test = {a: pearson_r_test[a][1:pearson_r_test[a].shape[0]:2] for a in pearson_r_test}
    pearson_r_late_test = {a: pearson_r_test[a][2:pearson_r_test[a].shape[0] + 1:2] for a in
                                 pearson_r_test}

    # Visualisations
    for a in pearson_r_early_test.keys():
        with open(main_summary_dir + '/pearson_early_test_' + a + '.csv',
                  'w', encoding='UTF8') as f:
            writer_file = csv.writer(f)
            writer_file.writerows(np.stack(pearson_r_early_test[a].numpy(), 0))

    for a in pearson_r_late_test.keys():
        with open(main_summary_dir + '/pearson_late_test_' + a + '.csv',
                  'w', encoding='UTF8') as f:
            writer_file = csv.writer(f)
            writer_file.writerows(np.stack(pearson_r_late_test[a].numpy(), 0))

    for a in pearson_r_early.keys():
        with open(main_summary_dir + '/pearson_early_' + a + '.csv', 'w',
                  encoding='UTF8') as f:
            writer_file = csv.writer(f)
            writer_file.writerows(pearson_r_early[a].numpy())

    for a in pearson_r_late.keys():
        with open(main_summary_dir + '/pearson_late_' + a + '.csv', 'w',
                  encoding='UTF8') as f:
            writer_file = csv.writer(f)
            writer_file.writerows(pearson_r_late[a].numpy())

    for a in pearson_r_early.keys():
         heatmap_early = HeatmapPlotter(main_summary_dir, a, "pearson_early_" + a)
         heatmap_late = HeatmapPlotter(main_summary_dir, a, "pearson_late_" + a)
         heatmap_early.create_heatmap()
         heatmap_late.create_heatmap()

    bars = BarPlotter(main_summary_dir, pearson_r_early.keys())
    bars.create_bar()

def convert_sequence_to_images(alphabet, sequence, main_labels, element='first'):
    if element == 'both':
        pairs_images = [torch.cat((alphabet[int(a[0])][0], alphabet[int(a[1])][0]), 2) for a in sequence]
    if element == 'first':
        pairs_images = [alphabet[int(a[0])][0] for a in sequence]
    if element == 'second':
        pairs_images = [alphabet[int(a[1])][0] for a in sequence]
    labels = [(alphabet[int(a[0])][1], alphabet[int(a[1])][1]) for a in sequence]
    labels = [main_labels.index(value) for value in labels]

    pairs_images = torch.stack(pairs_images, 0)

    return pairs_images, labels

def format_data(alphabet, data, labels, model,device, coefficient, option):
    paired_data, paired_target = convert_sequence_to_images(alphabet=alphabet,
                                                                        sequence=data, element='both',
                                                                        main_labels=labels)

    data_A, target = convert_sequence_to_images(alphabet=alphabet,
                                                            sequence=data,
                                                            element='first',
                                                            main_labels=labels)
    data_B, _ = convert_sequence_to_images(alphabet=alphabet,
                                                 sequence=data,
                                                 element='second',
                                                 main_labels=labels)
    target = torch.tensor(target, dtype=torch.long, device=device)

    data_A = data_A.to(device)
    data_B = data_B.to(device)

    _, data_A = model(data_A, target, mode='extractor')
    _, data_B = model(data_B, target, mode='extractor')

    data = torch.empty(data_A['ltm']['memory']['output'].size(), dtype=torch.float32)

    for i in range(data_A['ltm']['memory']['output'].size(0)):
        data_A_i = data_A['ltm']['memory']['output'].detach()[i]
        data_B_i = data_B['ltm']['memory']['output'].detach()[i]
        embedder = Overlap(coefficient, option)
        data_i = embedder.join(data_A_i, data_B_i)
        data[i] = data_i

    return data, target, paired_data

if __name__ == '__main__':
    main()
