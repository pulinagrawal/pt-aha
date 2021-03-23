"""oneshot_metrics.py"""

import torch
import numpy as np

import utils


class OneshotMetrics:
  """
  A self-contained module to compute, manage and report the metrics for one-shot learning.

  The metrics include standard metrics (accuracy, MSE) as well as similarity/matching metrics
  used in the Lake's Omniglot experiments.
  """

  def __init__(self):
    self.metrics = {}
    self.average_metrics = {}

  def update_averages(self, key, value):
    if key not in self.average_metrics.keys():
      self.average_metrics[key] = []

    self.average_metrics[key].append(value)

  def compare(self, prefix, primary_features, primary_labels, secondary_features, secondary_labels, comparison_type):
    # Accuracy based on similarity matrix
    if comparison_type.startswith('match_'):
      comparison_subtype = comparison_type.split('_')[1]  # mse or overlap

      if comparison_subtype not in ['mse', 'overlap']:
        raise NotImplementedError('Comparison type not supported: ' + comparison_type)

      similarity_matrix = utils.compute_matrix(primary_features, secondary_features, comparison_subtype)
      truth_matrix = utils.compute_truth_matrix(primary_labels, secondary_labels)

      match_accuracy, sum_ambiguous = utils.compute_accuracy(similarity_matrix, truth_matrix, comparison_subtype)

      self.metrics[prefix + '_' + 'acc_' + comparison_subtype] = match_accuracy
      self.metrics[prefix + '_' + 'amb_' + comparison_subtype] = sum_ambiguous

    # Accuracy based on labels
    elif comparison_type == 'accuracy':
      self.metrics[prefix + '_' + 'accuracy'] = torch.eq(primary_labels, secondary_labels).float().mean()

    # MSE
    elif comparison_type == 'mse':
      self.metrics[prefix + '_' + 'mse'] = utils.mse(primary_features, secondary_features)

    # Mismatch loss between two features
    elif comparison_type == 'mismatch':
      self.metrics[prefix + '_' + 'mismatch'] = torch.sum(torch.abs(secondary_features - primary_features)) / primary_features.shape[0]

    # Cosine similarity between two features
    elif comparison_type == 'cosine_similarity':
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
      self.metrics[prefix + '_' + 'cosine_similarity'] = cos(primary_features, secondary_features)

  def report(self, verbose=True):
    """Format and report specified metrics, and update running averages."""
    skip_console = []

    def log_to_console(x):
      if not verbose:
        return
      print(x)

    log_to_console("\n--------- Metrics -----------")
    np.set_printoptions(threshold=np.inf)

    for metric, metric_value in self.metrics.items():
      if metric not in skip_console:
        log_to_console("\t{0} : {1}".format(metric, metric_value))

        self.update_averages('{}'.format(metric), metric_value)

    # Averages updated; reset internal metrics
    self.metrics = {}

  def report_averages(self, export_csv=True):
    """Report the averaged metrics, and optionally export a CSV-friendly format."""
    if not self.average_metrics:
      return

    print("\n--------- Averages for all batches: ----------")
    for accuracy_type, vals in self.average_metrics.items():
      av = np.mean(vals, dtype=np.float64)
      print("\t{}: {}     (length={})".format(accuracy_type, av, len(vals)))

    print('\n')

    if export_csv:
      # print as comma separated list for import into a spreadsheet
      # headings
      for accuracy_type, vals in self.average_metrics.items():
        print("{}, ".format(accuracy_type), end='')
      print('\n')

      # values
      for accuracy_type, vals in self.average_metrics.items():
        av = np.mean(vals, dtype=np.float64)
        print("{}, ".format(av), end='')
      print('\n')
