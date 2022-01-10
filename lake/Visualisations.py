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

        for a in self.components:
            data_initial = self._load_predictions_pearson("pearson_initial_test_" + a)

            tmp_df_A = pd.DataFrame({'A': data_initial[:, 0], 'B': labels[0]})
            tmp_df_B = pd.DataFrame({'A': data_initial[:, 1], 'B': labels[1]})
            tmp_df_ini = pd.concat([tmp_df_A, tmp_df_B])
            tmp_df_ini['C'] = a
            data_settled = self._load_predictions_pearson("pearson_settled_test_" + a)

            tmp_df_A = pd.DataFrame({'A': data_settled[:, 0], 'B': labels[2]})
            tmp_df_B = pd.DataFrame({'A': data_settled[:, 1], 'B': labels[3]})
            tmp_df_shu = pd.concat([tmp_df_A, tmp_df_B])
            tmp_df_shu['C'] = a

            df = pd.concat([df, tmp_df_ini, tmp_df_shu])

        bar_plot = sns.barplot(x='C', y='A', hue='B', data=df, palette=['#DBAE81', '#B29EC1', '#D17A3D', '#685CA2'],
                               errwidth=0.5)
        box = bar_plot.get_position()
        bar_plot.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
        plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, frameon=False)
        bar_plot.set(xlabel=None)
        plt.ylabel("Mean correlation between \npatterns after training")
        figure = bar_plot.get_figure()
        figure.savefig(os.path.join(self.path, 'bar.png'), dpi=300)

# for a in ['dg', 'ca3', 'pr', 'ca3_ca1', 'ca1', 'final']:
#           heatmap_initial = HeatmapPlotter("/Users/karina/PycharmProjects/pt-aha/lake/runs/associative_inference/recurrence/20211217-171312/predictions", "pearson_initial_" + a)
#           heatmap_settled = HeatmapPlotter("/Users/karina/PycharmProjects/pt-aha/lake/runs/associative_inference/recurrence/20211217-171312/predictions", "pearson_settled_" + a)
#           heatmap_initial.create_heatmap()
#           heatmap_settled.create_heatmap()


#experiment = 'associative_inference'
experiment = 'pairs_structure'
#type = 'recurrence'
type = 'episodic'
date = '20220104-162033'
components = ['dg', 'ca3', 'ca3_ca1', 'ca1', 'pr']
path = '/Users/karina/PycharmProjects/pt-aha/lake/runs/'


bars = BarPlotter(os.path.join(path, experiment, type, date, 'predictions'), components)
bars.create_bar()

for a in components:
            heatmap_initial = HeatmapPlotter(os.path.join(path, experiment, type, date, 'predictions'), "pearson_initial_" + a)
            heatmap_settled = HeatmapPlotter(os.path.join(path, experiment, type, date, 'predictions'), "pearson_settled_" + a)
            heatmap_initial.create_heatmap()
            heatmap_settled.create_heatmap()