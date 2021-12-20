import csv
import os
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import torch


class HeatmapPlotter:

    def __init__(self, path, component):
        self.path = path
        self.component = component

    # Function to load results in Runs
    def _load_predictions_pearson(self):
        name = self.component + '.*?csv$'
        response = [file for file in os.listdir(self.path) if re.match(name, file)]
        li = []
        for filename in response:
            with open(os.path.join(self.path, filename), newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                data = [list(map(float, a)) for a in data]
                data = torch.FloatTensor(data)
            li.append(data)
        li = torch.stack(li)

        return li

    def create_heatmap(self):
        data = self._load_predictions_pearson()
        data_mean = torch.mean(data, 0)

        labels = [list(string.ascii_uppercase)[a] for a in range(list(data.size())[1])]
        plt.figure()
        ax = plt.axes()
        r_heatmap = sns.heatmap(data_mean, xticklabels=labels, yticklabels=labels, cmap="RdYlBu_r", ax=ax)
                               # vmin=-0.1, vmax=1, ax=ax)

        ax.set_title(self.component)
        figure = r_heatmap.get_figure()
        figure.savefig(os.path.join(self.path, self.component + '.png'), dpi=300)



for a in ['dg', 'ca3', 'pr', 'ca3_ca1', 'ca1', 'final']:
          heatmap_initial = HeatmapPlotter("/Users/karina/PycharmProjects/pt-aha/lake/runs/associative_inference/recurrence/20211217-171312/predictions", "pearson_initial_" + a)
          heatmap_settled = HeatmapPlotter("/Users/karina/PycharmProjects/pt-aha/lake/runs/associative_inference/recurrence/20211217-171312/predictions", "pearson_settled_" + a)
          heatmap_initial.create_heatmap()
          heatmap_settled.create_heatmap()
