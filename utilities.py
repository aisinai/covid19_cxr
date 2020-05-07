import csv
import h5py
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXrayHDF5(Dataset):
    def __init__(self, path):
        hdf5_database = h5py.File(path, 'r')
        self.path = path
        self.hdf5_database = hdf5_database

    def __getitem__(self, index):
        hdf5_image = self.hdf5_database["img"][index, ...]  # read image
        image = torch.from_numpy(hdf5_image)
        # returns numpy.ndarray
        score = torch.tensor(self.hdf5_database["scores"][index], dtype=torch.long)
        admit = torch.tensor(self.hdf5_database["admits"][index], dtype=torch.long)
        intub = torch.tensor(self.hdf5_database["intubs"][index], dtype=torch.long)
        death = torch.tensor(self.hdf5_database["deaths"][index], dtype=torch.long)
        return image, score, admit, intub, death

    def __len__(self):
        return self.hdf5_database["img"].shape[0]


class CXRDataset(Dataset):
    def __init__(self, img_dir, list_file, img_size, mode, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding scores.
            mode: 'frontal' or 'lateral'
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        scores = []
        admits = []
        intubs = []
        deaths = []

        f = open(list_file, 'r')
        d = ','
        reader = csv.reader(f, delimiter=d)
        next(reader, None)
        for row in reader:
            image_name = os.path.join(img_dir, f"{row[0].zfill(4)}.jpg")
            image_names.append(image_name)
            score = float(row[1])
            scores.append(score)
            admit = int(row[2])
            admits.append(admit)
            intub = int(row[3])
            intubs.append(intub)
            death = int(row[4])
            deaths.append(death)

        self.image_names = image_names
        self.scores = scores
        self.admits = admits
        self.intubs = intubs
        self.deaths = deaths
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            padded image and its clinical variables
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            image = image * std + mean
        score = self.scores[index]
        admit = self.admits[index]
        intub = self.intubs[index]
        death = self.deaths[index]

        return image, score, admit, intub, death

    def __len__(self):
        return len(self.image_names)


class densenet_last_layer(torch.nn.Module):
    def __init__(self, model):
        super(densenet_last_layer, self).__init__()
        self.features = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.relu(x, inplace=True)
        return x


def compute_auroc(gt, pred, save_path, index):
    font = {'size': 20}
    plt.rc('font', **font)
    labels = ['Severity', 'Admission', 'Intubation', 'Death']
    fpr, tpr, thresholds = metrics.roc_curve(gt, pred)
    auroc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(15, 15))
    plt.plot(fpr, tpr, color='darkorange',
             label=f'ROC curve (area = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{labels[index]} Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_path}/auroc_{labels[index]}.png')
    plt.close()

    return auroc


def save_loss_plot(n_epochs, latest_epoch, losses, save_path):
    font = {'size': 20}

    plt.rc('font', **font)
    epochs = range(1, n_epochs + 1)
    fig = plt.figure(figsize=(15, 15))
    latest_epoch += 1
    if latest_epoch > 150:
        latest_epoch = 150

    fontsize = 16
    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, losses[0, :], '-')
    ax1.plot(epochs, losses[1, :], '-')
    ax1.set_title('All Losses')
    ax1.set_xlabel('Epochs', fontsize=fontsize)
    ax1.axis(xmin=1, xmax=latest_epoch)
    ax1.legend(["Train Loss", "Validation Loss"], loc="upper right")

    plt.close(fig)
    fig.savefig(f'{save_path}/loss graphs.png')


def save_auroc_plot(n_epochs, latest_epoch, aurocs, save_path):
    font = {'size': 20}
    plt.rc('font', **font)
    epochs = range(1, n_epochs + 1)
    fig = plt.figure(figsize=(15, 15))
    latest_epoch += 1
    if latest_epoch > 150:
        latest_epoch = 150

    fontsize = 16
    ymin = 0
    ax1 = fig.add_subplot(221)
    ax1.plot(epochs, aurocs[0, :, 0], '-')
    ax1.plot(epochs, aurocs[1, :, 0], '-')
    ax1.legend(["Train", "Val"], loc="lower right")
    ax1.set_title('AUROC CXR Severity Score')
    ax1.set_xlabel('Epochs', fontsize=fontsize)
    ax1.axis(xmin=0, xmax=latest_epoch, ymin=ymin, ymax=1)

    ax2 = fig.add_subplot(222)
    ax2.plot(epochs, aurocs[0, :, 1], '-')
    ax2.plot(epochs, aurocs[1, :, 1], '-')
    ax2.legend(["Train", "Val"], loc="lower right")
    ax2.set_title('AUROC Admissions')
    ax2.set_xlabel('Epochs', fontsize=fontsize)
    ax2.axis(xmin=0, xmax=latest_epoch, ymin=ymin, ymax=1)

    ax3 = fig.add_subplot(223)
    ax3.plot(epochs, aurocs[0, :, 2], '-')
    ax3.plot(epochs, aurocs[1, :, 2], '-')
    ax3.legend(["Train", "Val"], loc="lower right")
    ax3.set_title('AUROC Intubations')
    ax3.set_xlabel('Epochs', fontsize=fontsize)
    ax3.axis(xmin=0, xmax=latest_epoch, ymin=ymin, ymax=1)

    ax4 = fig.add_subplot(224)
    ax4.plot(epochs, aurocs[0, :, 3], '-')
    ax4.plot(epochs, aurocs[1, :, 3], '-')
    ax4.legend(["Train", "Val"], loc="lower right")
    ax4.set_title('AUROC Deaths')
    ax4.set_xlabel('Epochs', fontsize=fontsize)
    ax4.axis(xmin=0, xmax=latest_epoch, ymin=ymin, ymax=1)

    plt.close(fig)
    fig.savefig(f'{save_path}/auroc graphs.png')
