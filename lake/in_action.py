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
    layout = [[sg.Text("Oh Lord, show me the way!")],
              [sg.Button("OK")],
              #[sg.Canvas(key="-CANVAS-")],
              [sg.Canvas(key="input"+str(i)) for i in range(5)],
              [sg.Canvas(key="input"+str(5+i)) for i in range(5)]
             ]

    # Create the window

    def onObjectClick(event):                  
        print('Got object click', event.x, event.y)
        print(event.widget.find_closest(event.x, event.y))
    
    window = sg.Window("Demo", layout, finalize=True)
    for canvases in layout[2:]:
        for canv in canvases:
            canv.tk_canvas.tag_bind(canv, '<ButtonPress-1>', onObjectClick)       

    x, _ = next(iter(study_loader))
    for i in range(10):
        fig = plt.figure(figsize=(1,1))
        plt.imshow(x[i+5].squeeze(), aspect='equal')
        plt.axis('off') 
        draw_figure(window['input'+str(i)].TKCanvas, fig)

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "OK" or event == sg.WIN_CLOSED:
            break

    window.close()

# %%



