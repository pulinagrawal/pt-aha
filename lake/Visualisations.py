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
        if 'mirror' in response[0]:
            mirror = True
        else:
            mirror = False

        return li, mirror

    def create_heatmap(self):
        data, mirror = self._load_predictions_pearson()
        data_mean = torch.mean(data, 0)
        if mirror:
            letters = [list(string.ascii_uppercase)[a] for a in range(list(data.size())[1] // 2)]
            first_letters = [a + "_" for a in letters]
            second_letters = ["_" + a for a in letters]
            labels = first_letters + second_letters
        else:
            letters = [list(string.ascii_uppercase)[a] for a in range(list(data.size())[1])]
            labels = [a + "_" for a in letters]
        plt.figure()
        ax = plt.axes()
        r_heatmap = sns.heatmap(data_mean, xticklabels=labels, yticklabels=labels, cmap="coolwarm",
                                vmin=-0.1, vmax=1, ax=ax)
        ax.set_title(self.component)
        figure = r_heatmap.get_figure()
        figure.savefig(os.path.join(self.path, self.component + '.png'), dpi=300)


# Calculate probability of producing items
class LinePlotter:

    def __init__(self, path, component):
        self.path = path
        self.component = component

        # Function to load results in Runs

    def _load_frequencies(self):
        name = self.component + '.*?csv$'
        response = [file for file in os.listdir(self.path) if re.match(name, file)]
        li = []
        for filename in response:
            with open(os.path.join(self.path, filename), newline='') as f:
                reader = csv.reader(f)
                data = [list(row) for row in reader]
            li.append(data)
        if 'mirror' in response[0]:
            mirror = True
        else:
            mirror = False
        return li, mirror

    # Function to calculate probability of producing items
    def plot_probability(self):
        data, mirror = self._load_frequencies()
        network = []
        epoch = []
        category = []
        value = []

        for i in range(len(data)):
            for j in range(len(data[i])):
                sumAtoA = 0
                sumAtoB = 0
                sumBtoB = 0
                sumBtoA = 0
                incorrect = 0
                for k in range(len(data[i][j])):
                    if k % 2 == 0:
                        error = True
                        if str(k) in data[i][j][k]:
                            sumAtoA = sumAtoA + 1
                            error = False
                        if str(k + 1) in data[i][j][k]:
                            sumAtoB = sumAtoB + 1
                            error = False
                        if error:
                            incorrect = incorrect + 1
                    else:
                        error = True
                        if str(k) in data[i][j][k]:
                            sumBtoB = sumBtoB + 1
                            error = False
                        if str(k - 1) in data[i][j][k]:
                            sumBtoA = sumBtoA + 1
                            error = False
                        if error:
                            incorrect = incorrect + 1
                network = network + [i] * 5
                epoch = epoch + [j] * 5
                category = category + ['AtoA', 'AtoB', 'BtoB', 'BtoA', 'Incorrect']
                value = value + [sumAtoA / ((k + 1) / 2), sumAtoB / ((k + 1) / 2), sumBtoB / ((k + 1) / 2),
                                 sumBtoA / ((k + 1) / 2), incorrect / ((k + 1) / 2)]
            d = {'network': network, 'epoch': epoch, 'category': category, 'value': value}

            plt.figure()
            line_plot = sns.lineplot(data=d, x="epoch", y="value", hue="category", err_style="bars", style='category',
                                     palette=['black', 'green', 'dimgrey', 'yellow', 'silver'])
            figure = line_plot.get_figure()
            figure.savefig(os.path.join(self.path, self.component + '.png'), dpi=300)
