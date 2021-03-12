# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import datasets, transforms
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import base64
import numpy as np

from omniglot_one_shot_dataset import OmniglotTransformation, OmniglotOneShotDataset

from cls_module.memory.ltm.visual_component import VisualComponent
from cls_module.memory.stm.aha import AHA
from cls_module.cls import CLS


import PySimpleGUI as sg

matplotlib.use("TkAgg")

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

# %%

with open('/mnt/c/Users/pulin/Projects/ProjectAGI/pt-aha/lake/definitions/aha_config.json') as config_file:
  config = json.load(config_file)

batch_size = 20
image_tfms = transforms.Compose([
    transforms.ToTensor(),
    OmniglotTransformation(resize_factor=0.5)
])


# %%
study_loader = torch.utils.data.DataLoader(
    OmniglotOneShotDataset('./data', train=True, download=True,
                           transform=image_tfms, target_transform=None),
    batch_size=batch_size, shuffle=False)

x, _ = next(iter(study_loader))


# %%
pretrained_model_path = '/mnt/c/Users/pulin/Projects/ProjectAGI/pt-aha/lake/runs/20210223-234348/pretrained_model_10.pt'

# %%
image_shape = config['image_shape']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

summary_dir = Path('./runs/in_action')
summary_dir.mkdir(exist_ok=True)
writer = SummaryWriter(log_dir=summary_dir)

model = CLS(image_shape, config, device=device, writer=writer).to(device)
model.load_state_dict(torch.load(pretrained_model_path))
model.reset()


# %%
'''
from matplotlib import pyplot as plt
for study in study_loader:
    print(study[0].shape)
    print(study[1].shape)
    for i in range(study[0].shape[0]):
        image = study[0][i,:]
        target = study[1][i]
        plt.imshow(image.squeeze())
        #model(image.unsqueeze(0), target.unsqueeze(0), mode='study')
        inp = input()
        if inp == 'q':
            break
        plt.show()
    if inp == 'q':
        break
'''


if __name__ == '__main__':

    background = '#F0F0F0'
    sg.set_options(background_color=background,
                   element_background_color=background)
    layout = [[sg.Text("Oh Lord, show me the way!")], [sg.Button("OK")],
              #[sg.Canvas(key="-CANVAS-")],
              [sg.Button(key="input"+str(i), button_color=(background, background), border_width=0)
                 for i in range(10)],
              []
             ]

    # Create the window
    window = sg.Window("Demo", layout, finalize=True)

    for i in range(10):
        x, _ = next(iter(study_loader))
        array = x[i].squeeze().numpy()
        array = np.stack((array, array, array)).transpose()

        image = cv2.imencode('.png', array*255)[1]
        image = base64.b64encode(image.tobytes()).decode('utf-8')
        window['input'+str(i)].update(image_data=image)

        #draw_figure(window['input'+str(i)].TKCanvas, fig)

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "OK" or event == sg.WIN_CLOSED:
            break

    window.close()

# %%



