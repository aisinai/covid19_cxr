import argparse
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from utilities import ChestXrayHDF5, densenet_last_layer

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument("--train_mode", type=str, default="Severity Score")
parser.add_argument("--index", type=int)
args = parser.parse_args()

model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

if args.train_mode == "Severity Score":
    model = torch.load("/home/aisinai/work/covid19/densenet121/20200507/score3/best_densenet_145.pt")
elif args.train_mode == "Admission Status":
    model = torch.load("/home/aisinai/work/covid19/densenet121/20200507/admit3/best_densenet_129.pt")

model = model.cuda() if cuda else model

model_cam = densenet_last_layer(model)
dataset = ChestXrayHDF5(f'/home/aisinai/work/HDF5_datasets/COVID19_binary_test_1024.hdf5')

font = {'size': 14}
plt.rc('font', **font)

args.index
img, score, admit, intub, death = dataset[args.index]
sample_img = torch.rand((1, 3, 1024, 1024))
sample_img[0, :] = img
sample_img = sample_img.cuda()

model_cam = densenet_last_layer(model)
x = torch.autograd.Variable(sample_img)
y = model_cam(x)
y = y.cpu().data.numpy()
y = np.squeeze(y)

weights = model.state_dict()['classifier.0.weight']
weights = weights.cpu().numpy()

bias = model.state_dict()['classifier.0.bias']
bias = bias.cpu().numpy()

cam = np.zeros((7, 7, 1))
for i in range(0, 7):
    for j in range(0, 7):
        for k in range(0, 1024):
            cam[i, j] += y[k, i, j] * weights[0, k]
cam += bias[0]
cam = 1 / (1 + np.exp(-cam))
cam = cam / (7 / 15)
cam = np.log(cam)

pred = model(torch.autograd.Variable(sample_img)).data[0]
pred = pred.detach().cpu().numpy()

fig, (showcxr, heatmap) = plt.subplots(ncols=2, figsize=(20, 7.14))

hmap = sns.heatmap(cam.squeeze(),
                   cmap='viridis',
                   alpha=0.3,  # whole heatmap is translucent
                   annot=True,
                   zorder=2, square=True, vmin=-1, vmax=1)

cxr = sample_img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
cxr = std * cxr + mean
cxr = np.clip(cxr, 0, 1)
r, g, b = cxr[:, :, 0], cxr[:, :, 1], cxr[:, :, 2]
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

hmap.imshow(cxr,
            aspect=hmap.get_aspect(),
            extent=hmap.get_xlim() + hmap.get_ylim(),
            zorder=1)  # put the map under the heatmap
hmap.axis('off')
hmap.set_title(f"P(Severity score) = {pred[0]:.4f}", fontsize=22)

showcxr.imshow(gray, cmap='gray')
showcxr.axis('off')
showcxr.set_title(f"Score: {score} Admission: {admit} Intubation: {intub} Death: {death}", fontsize=22)
plt.show()
plt.savefig(f"heatmap_{args.index}.png")
plt.close()
