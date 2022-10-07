import csv
import os
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import collections

import seaborn as sns
import pandas as pd

COMPONENT_NAME_MAP = {
  "dg": "DG",
  "ca3": "CA3",
  "pr": "EC:CA3 (Err)",
  "ec_ca3": "EC:CA3 (Hebb)",
  "ca3_ca1": "CA3:CA1",
  "ca1_enc": "CA1",
  "ca1_dec": "ECout",
  "recon_pair": "Reconstruction"
}

class HeatmapPlotter:
    def __init__(self, path, component, label):
        self.path = path
        self.component = component
        self.label = label
        self.community = "community" in path

    # Function to load results in Runs
    def _load_predictions_pearson(self):
        name = self.label + '.csv$'
        response = [file for file in os.listdir(self.path) if re.match(name, file)]
        with open(os.path.join(self.path, response[0]), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            data = [list(map(float, a)) for a in data]
            data = np.stack(data, axis=0)

        return data

    def create_heatmap(self):
        data = self._load_predictions_pearson()

        labels = [list(string.ascii_uppercase)[a] for a in range(data.__len__())]
        plt.figure()
        ax = plt.axes()

        if not self.community:
            r_heatmap = sns.heatmap(data, xticklabels=labels, yticklabels=labels, cmap="RdYlBu_r",
                                vmin=-0.1, vmax=1, ax=ax)
        else:
            r_heatmap = sns.heatmap(data, xticklabels=labels, yticklabels=labels, cmap="RdYlBu_r",
                                    vmin=-0.1, vmax=0.6, ax=ax)

        ax.set_title(COMPONENT_NAME_MAP[self.component])
        figure = r_heatmap.get_figure()
        figure.savefig(os.path.join(self.path, self.label + '.png'), dpi=300)
        plt.close()


class BarPlotter:
    def __init__(self, path, components):
        self.path = path
        self.components = components

    def _load_predictions_pearson(self, component):
        name = component + '.*?csv$'
        response = [file for file in os.listdir(self.path) if re.match(name, file)]
        with open(os.path.join(self.path, response[0]), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            data = [list(map(float, a)) for a in data]
            data = np.stack(data, axis=0)

        return data

    def create_bar(self):
        df = pd.DataFrame({'A': [], 'B': [], 'C': []})

        if 'pairs_structure' in self.path:
             labels = ['Early \npair', 'Early \nshuffled', 'Late \npair', 'Late \nshuffled']
        if 'associative_inference' in self.path:
            labels = ['Early \ntransitive', 'Early \ndirect', 'Late \ntransitive', 'Late \ndirect']
        if 'community_structure' in self.path:
            labels = ['Early \nwithin internal', 'Early \nwithin boundary',
                      'Early \nacross boundary', 'Early \nacross other',
                      'Late \nwithin internal', 'Late \nwithin boundary',
                      'Late \nacross boundary', 'Late \nacross other']

        for a in self.components:
            data_early = self._load_predictions_pearson("pearson_early_test_" + a)

            tmp_df_A = pd.DataFrame({'A': data_early[:, 0], 'B': labels[0]})
            tmp_df_B = pd.DataFrame({'A': data_early[:, 1], 'B': labels[1]})

            tmp_df_ini = pd.concat([tmp_df_A, tmp_df_B])

            if 'community_structure' in self.path:
                tmp_df_C = pd.DataFrame({'A': data_early[:, 2], 'B': labels[2]})
                tmp_df_D = pd.DataFrame({'A': data_early[:, 3], 'B': labels[3]})
                tmp_df_ini = pd.concat([tmp_df_A, tmp_df_B, tmp_df_C, tmp_df_D])

            tmp_df_ini['C'] = a

            data_late = self._load_predictions_pearson("pearson_late_test_" + a)

            tmp_df_A = pd.DataFrame({'A': data_late[:, 0], 'B': labels[2]})
            tmp_df_B = pd.DataFrame({'A': data_late[:, 1], 'B': labels[3]})
            tmp_df_shu = pd.concat([tmp_df_A, tmp_df_B])

            if 'community_structure' in self.path:
                tmp_df_A = pd.DataFrame({'A': data_late[:, 0], 'B': labels[4]})
                tmp_df_B = pd.DataFrame({'A': data_late[:, 1], 'B': labels[5]})
                tmp_df_C = pd.DataFrame({'A': data_late[:, 2], 'B': labels[6]})
                tmp_df_D = pd.DataFrame({'A': data_late[:, 3], 'B': labels[7]})
                tmp_df_shu = pd.concat([tmp_df_A, tmp_df_B, tmp_df_C, tmp_df_D])

            tmp_df_shu['C'] = a

            df = pd.concat([df, tmp_df_ini, tmp_df_shu])

        bar_colors = ['#DBAE81', '#B29EC1', '#D17A3D', '#685CA2']

        if 'community_structure' in self.path:
            bar_colors = ['#B995C2', '#EAB264', '#C1E0EE', '#87AD57', '#4F2C8A', '#D85925', '#364BAC', '#3A7529']

        bar_plot = sns.barplot(x='C', y='A', hue='B', data=df, palette=bar_colors,
                               errwidth=0.5)
        box = bar_plot.get_position()
        bar_plot.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
        plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, frameon=False)
        bar_plot.set(xlabel=None)
        plt.ylabel("Mean correlation between \npatterns after training")
        figure = bar_plot.get_figure()
        figure.savefig(os.path.join(self.path, 'bar.png'), dpi=300)
        plt.close()

class FrequencyPlotter:
    def __init__(self, path, sequence, seed_batch):
        self.path = path
        self.sequence = sequence.tolist()
        self.seed_batch = seed_batch

    def create_bar(self):
        plt.clf()
        characters = [list(string.ascii_uppercase)[int(a)] + list(string.ascii_uppercase)[int(b)] for [a, b] in self.sequence]
        frequency = collections.Counter(characters)
        frequency = {k: v for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}
        plt.bar(frequency.keys(), frequency.values())
        y = [i for i in range(0, max(frequency.values())+1, 2)]
        plt.yticks(y, y)
        plt.grid(axis='y')
        plt.savefig(os.path.join(self.path, 'frequency_' + self.seed_batch + '.png'), dpi=300)
        plt.close()
