import csv
import os
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import numpy as np


class HeatmapPlotter:

    def __init__(self, path, component):
        self.path = path
        self.component = component

    # Function to load results in Runs
    def _load_predictions_pearson(self):
        name = self.component + '.*?csv$'
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
        r_heatmap = sns.heatmap(data, xticklabels=labels, yticklabels=labels, cmap="RdYlBu_r",
                                vmin=-0.1, vmax=1, ax=ax)

        ax.set_title(self.component)
        figure = r_heatmap.get_figure()
        figure.savefig(os.path.join(self.path, self.component + '.png'), dpi=300)


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
             labels = ['Initial \npair', 'Initial \nshuffled', 'Settled \npair', 'Settled \nshuffled']
        if 'associative_inference' in self.path:
            labels = ['Initial \ntransitive', 'Initial \ndirect', 'Settled \ntransitive', 'Settled \ndirect']
        if 'community_structure' in self.path:
            labels = ['Initial \nwithin internal', 'Initial \nwithin boundary',
                      'Initial \nacross boundary', 'Initial \nacross other',
                      'Settled \nwithin internal', 'Settled \nwithin boundary',
                      'Settled \nacross boundary', 'Settled \nacross other']

        for a in self.components:
            data_initial = self._load_predictions_pearson("pearson_initial_test_" + a)

            tmp_df_A = pd.DataFrame({'A': data_initial[:, 0], 'B': labels[0]})
            tmp_df_B = pd.DataFrame({'A': data_initial[:, 1], 'B': labels[1]})

            tmp_df_ini = pd.concat([tmp_df_A, tmp_df_B])

            if 'community_structure' in self.path:
                tmp_df_C = pd.DataFrame({'A': data_initial[:, 2], 'B': labels[2]})
                tmp_df_D = pd.DataFrame({'A': data_initial[:, 3], 'B': labels[3]})
                tmp_df_ini = pd.concat([tmp_df_A, tmp_df_B, tmp_df_C, tmp_df_D])

            tmp_df_ini['C'] = a

            data_settled = self._load_predictions_pearson("pearson_settled_test_" + a)

            tmp_df_A = pd.DataFrame({'A': data_settled[:, 0], 'B': labels[2]})
            tmp_df_B = pd.DataFrame({'A': data_settled[:, 1], 'B': labels[3]})
            tmp_df_shu = pd.concat([tmp_df_A, tmp_df_B])

            if 'community_structure' in self.path:
                tmp_df_A = pd.DataFrame({'A': data_settled[:, 0], 'B': labels[4]})
                tmp_df_B = pd.DataFrame({'A': data_settled[:, 1], 'B': labels[5]})
                tmp_df_C = pd.DataFrame({'A': data_settled[:, 2], 'B': labels[6]})
                tmp_df_D = pd.DataFrame({'A': data_settled[:, 3], 'B': labels[7]})
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
