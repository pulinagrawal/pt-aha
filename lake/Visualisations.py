import os, re
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import csv
experiment = 'pairs_structure'
type = 'statistical'
date = '20210824-181923'

#all_filenames = [i for i in glob.glob('*.{}'.format('csv'))]

path = os.path.join('runs', experiment, type, 'predictions', date)


# Function to load results in Runs
def load_predictions_pearson(file_name, path):
    name = file_name + '.*?csv$'
    response = [file for file in os.listdir(path) if re.match(name, file)]
    li = []
    for filename in response:
        with open(os.path.join(path, filename), newline='') as f:
            reader = csv.reader(f)
            if 'pearson' in file_name:
                data = list(reader)
                data = [list(map(float, a)) for a in data]
                data = torch.FloatTensor(data)
            else:
                data = [list(row) for row in reader]
        li.append(data)
    if 'pearson' in file_name:
        li = torch.stack(li)

    return li

# Load results in Runs
initial_response = load_predictions_pearson("predictions_initial", path)
settled_response = load_predictions_pearson("predictions_settled", path)
pearsonr_initial = load_predictions_pearson("pearsonr_initial", path)
pearsonr_settled = load_predictions_pearson("pearsonr_settled", path)

# Plot r scores (Pearson correlation)
pearsonr_initial_mean = torch.mean(pearsonr_initial, 0)
pearsonr_settled_mean = torch.mean(pearsonr_settled, 0)

r_initial = sns.heatmap(pearsonr_initial_mean, cmap="coolwarm", vmin=-0.1, vmax=1)
figure = r_initial.get_figure()
figure.savefig(os.path.join(path, 'r_initial.png'), dpi=300)

plt.figure()
r_settled = sns.heatmap(pearsonr_settled_mean, cmap="coolwarm", vmin=-0.1, vmax=1)
figure2 = r_settled.get_figure()
figure2.savefig(os.path.join(path, 'r_settled.png'), dpi=300)


# Function to calculate probability of producing items
def get_probability(data):
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
                    if str(k+1) in data[i][j][k]:
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
            network = network + [i]*5
            epoch = epoch + [j]*5
            category = category + ['AtoA', 'AtoB', 'BtoB', 'BtoA', 'Incorrect']
            value = value + [sumAtoA/((k+1)/2), sumAtoB/((k+1)/2), sumBtoB/((k+1)/2), sumBtoA/((k+1)/2), incorrect/((k+1)/2)]
            #value = value + [sumAtoA, sumAtoB, sumBtoB, sumBtoA, incorrect]
    d = {'network': network, 'epoch': epoch, 'category': category, 'value': value}

    return d


prob_initial = get_probability(initial_response)
prob_settled = get_probability(settled_response)

# Plot probabilities
plt.figure()
prob_ini = sns.lineplot(data=prob_initial, x="epoch", y="value", hue="category", err_style="bars", style='category',
             palette=['black', 'green', 'dimgrey', 'yellow', 'silver'])
figure = prob_ini.get_figure()
figure.savefig(os.path.join(path, 'prob_initial.png'), dpi=300)

plt.figure()
prob_ini = sns.lineplot(data=prob_settled, x="epoch", y="value", hue="category", err_style="bars", style='category',
             palette=['black', 'green', 'dimgrey', 'yellow', 'silver'])
figure = prob_ini.get_figure()
figure.savefig(os.path.join(path, 'prob_settled.png'), dpi=300)
