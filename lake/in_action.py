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

def setup_aha(config_filepath, pretrained_model_path):
    with open(config_filepath) as config_file:
      config = json.load(config_file)

    batch_size = 20
    image_tfms = transforms.Compose([
        transforms.ToTensor(),
        OmniglotTransformation(resize_factor=0.5)
    ])

    study_loader = torch.utils.data.DataLoader(
        OmniglotOneShotDataset('./data', train=True, download=True,
                               transform=image_tfms, target_transform=None),
        batch_size=batch_size, shuffle=False)

    image_shape = config['image_shape']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    summary_dir = Path('./runs/in_action')
    summary_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=summary_dir)

    model = CLS(image_shape, config, device=device, writer=writer).to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.reset()
    return study_loader, model



def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()

def update_image(window_object, x, title=''):
    fig = plt.figure(figsize=(2,2))
    plt.imshow(x, aspect='equal')
    plt.title(title)
    plt.axis('off') 
    return draw_figure(window_object.TKCanvas, fig)


if __name__ == '__main__':
    '''
    Usage:
            Please provide the absolute path to the model config file and pth file of the 
            pre-trained model.

            You can simply run this file with python
            $ python in_action.py
    '''
    config_filepath = '/mnt/c/Users/pulin/Projects/ProjectAGI/pt-aha/lake/definitions/aha_config.json'
    pretrained_model_path = '/mnt/c/Users/pulin/Projects/ProjectAGI/pt-aha/lake/runs/20210223-234348/pretrained_model_10.pt'

    study_loader, model = setup_aha(config_filepath, pretrained_model_path)
    data_iterator = iter(study_loader)
    x, label = next(data_iterator)

    # Setup GUI
    fig, ax = plt.subplots()
    background = '#F0F0F0'
    sg.set_options(background_color=background,
                   element_background_color=background)
    
    canvases = ['input', 'stm_ps', 'stm_pr', 'stm_pc', 'stm_recon', 'ltm_recon']
    layout = [[sg.Text("Show me what to remember and I shall.")],
              [sg.Button(key=str(i), button_color=(background, background), border_width=0)
                 for i in range(10)],
              [sg.Canvas(key=cnv_name) for cnv_name in canvases],
              [sg.Button('Study', disabled=True), sg.Button('Recall'), sg.Button('Change Set')],
             ]

    # Create the window
    window = sg.Window("Demo", layout, finalize=True)

    # Create an event loop
    mode = 'study'
    prev_fig_agg = [] 
    while True:

        for i in range(10):
            array = x[i].squeeze().numpy().transpose()
            array = np.stack((array, array, array)).transpose()

            image = cv2.imencode('.png', array*255)[1]
            image = base64.b64encode(image.tobytes()).decode('utf-8')
            window[str(i)].update(image_data=image)


        event, values = window.read()
        fig_agg = []
        # End program if user closes window 
        if event == 'Change Set':
            x, label = next(data_iterator)
        elif event == 'Study':
            window['Study'].update(disabled=True)
            window['Recall'].update(disabled=False)
            mode = 'study'
        elif event == 'Recall':
            window['Study'].update(disabled=False)
            window['Recall'].update(disabled=True)
            mode = 'recall'
        else:
            button = int(event) 
            for fagg in prev_fig_agg:
                delete_figure_agg(fagg)
            model(x[button].unsqueeze(0), label[button].unsqueeze(0), mode=mode)
            fig_agg.append(update_image(window['input'], x[button].squeeze(), 'input'))
            for feature in model.features[mode]:
                print(feature, model.features[mode][feature].shape)
                if feature not in ['inputs', 'labels', 'ltm_vc']:
                    if len(model.features[mode][feature].squeeze().shape)<2:
                        size = model.features[mode][feature].squeeze().shape[0]
                        shape = (int(size**.5), int(size**.5))
                        print(shape)
                        disp = model.features[mode][feature].reshape(shape)
                    else:
                        disp = model.features[mode][feature].squeeze()
                    fig_agg.append(update_image(window[feature], disp, feature))
            prev_fig_agg = fig_agg
            
    window.close()




