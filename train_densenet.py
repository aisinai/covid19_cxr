import argparse
import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm
from utilities import ChestXrayHDF5, compute_auroc, save_loss_plot, save_auroc_plot

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--n_classes', type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-6, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="adam: weight decay (L2 penalty)")
parser.add_argument('--data_path', type=str, default="/home/aisinai/work/HDF5_datasets")
parser.add_argument('--save_path', type=str, default="/home/aisinai/work/covid19/densenet121/20200717")
parser.add_argument('--mode', type=str, default="score")
parser.add_argument('--train_run', type=str, default="score3")
args = parser.parse_args()
print(args)
torch.manual_seed(816)

save_path = f'{args.save_path}/{args.train_run}'
os.makedirs(save_path, exist_ok=True)
with open(f'{save_path}/args.txt', 'w') as f:
    for key in vars(args).keys():
        f.write(f'{key}: {vars(args)[key]}\n')
        print(f'{key}: {vars(args)[key]}')

model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
model = model.cuda() if cuda else model
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    device_ids = list(range(n_gpu))
    model = nn.DataParallel(model, device_ids=device_ids)

dataloaders = {}
dataloaders['train'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/COVID19_binary_train_{args.size}.hdf5'),
                                  batch_size=4,
                                  shuffle=True,
                                  drop_last=True)
dataloaders['valid'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/COVID19_binary_valid_{args.size}.hdf5'),
                                  batch_size=4,
                                  shuffle=False,
                                  drop_last=False)

lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

losses = np.zeros((2, args.n_epochs))  # [0,:] for train, [1,:] for val
aurocs = np.zeros((2, args.n_epochs, 4))  # [0,:] for train, [1,:] for val
best_loss = 999999

for epoch in range(args.n_epochs):
    for phase in ['train', 'valid']:
        model.train(phase == 'train')
        loader = tqdm(dataloaders[phase])
        scores = torch.LongTensor()
        admits = torch.LongTensor()
        intubs = torch.LongTensor()
        deaths = torch.LongTensor()
        preds = torch.FloatTensor().cuda() if cuda else torch.FloatTensor()

        for i, (img, score, admit, intub, death) in enumerate(loader):
            scores = torch.cat((scores, score), 0)
            admits = torch.cat((admits, admit), 0)
            intubs = torch.cat((intubs, intub), 0)
            deaths = torch.cat((deaths, death), 0)
            real_img = Variable(img.type(Tensor))
            if args.mode == 'score':
                real_targets = Variable(score.cuda()).float() if cuda else Variable(score).float()
            else:
                real_targets = Variable(admit.cuda()).float() if cuda else Variable(score).float()
            with torch.set_grad_enabled(phase == 'train'):
                output_C = torch.squeeze(model(real_img))
                preds = torch.cat((preds, output_C), 0)
                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                c_loss = criterion(output_C, real_targets)
                if phase == 'train':
                    c_loss.backward()
                    optimizer.step()
            loader.set_description((f'phase: {phase}; epoch: {epoch + 1};'
                                    f'total_loss: {c_loss.item():.4f}; lr: {lr:.5f}'))

        preds = preds.detach().cpu()
        if phase == 'train':
            losses[0, epoch] = c_loss
            index = 0
            for comparison in [scores, admits, intubs, deaths]:
                aurocs[0, epoch, index] = compute_auroc(comparison, preds, save_path, index)
                index += 1
            print(f"{aurocs[0, epoch, :]}")

        elif phase == 'valid':
            losses[1, epoch] = c_loss
            index = 0
            for comparison in [scores, admits, intubs, deaths]:
                aurocs[1, epoch, index] = compute_auroc(comparison, preds, save_path, index)
                index += 1
            if c_loss < best_loss:
                best_loss = c_loss
                torch.save(model, f'{save_path}/best_densenet_{str(epoch + 1).zfill(3)}.pt')

        save_loss_plot(args.n_epochs, epoch, losses, save_path)
        save_auroc_plot(args.n_epochs, epoch, aurocs, save_path)

    torch.save(losses, f'{save_path}/losses.pt')
    torch.save(aurocs, f'{save_path}/aurocs.pt')
