import argparse
import h5py
import os
from torchvision import transforms
from utilities import CXRDataset


parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=1024)
parser.add_argument("--crop_size", type=int, default=1024)
args = parser.parse_args()

IMG_DIR = "/home/aisinai/work/covid19/images"
DATA_DIR = "/home/aisinai/work/covid19"
HDF5_DIR = "/home/aisinai/work/HDF5_datasets"
os.makedirs(HDF5_DIR, exist_ok=True)

nc = 3  # number of channels; 3 for RGB, 1 for grayscale
mean = [0.485, 0.456, 0.406]  # ImageNet mean
std = [0.229, 0.224, 0.225]  # ImageNet std
normalization = transforms.Normalize(mean=mean, std=std)
transform_array = [transforms.Resize(args.img_size),
                   transforms.CenterCrop(args.crop_size),
                   transforms.ToTensor(),
                   normalization]

# Generate HDF5 dataset
# for mode in ["train", "valid", "test"]:
for mode in ["test"]:
    list_file = os.path.join(DATA_DIR, f"{mode}.csv")
    dataset = CXRDataset(img_dir=IMG_DIR,
                         list_file=list_file,
                         img_size=args.img_size,
                         mode=mode,
                         transform=transforms.Compose(transform_array))

    num_images = len(dataset)  # total number of images in train set
    shape = (num_images, nc, args.crop_size, args.crop_size)
    hdf5_file_name = f"COVID19_binary_{mode}_{args.crop_size}.hdf5"
    hdf5_file_path = os.path.join(HDF5_DIR, hdf5_file_name)
    hdf5 = h5py.File(hdf5_file_path, "w")
    hdf5.create_dataset("scores", (num_images,), dtype='i')
    hdf5.create_dataset("admits", (num_images,), dtype='i')
    hdf5.create_dataset("intubs", (num_images,), dtype='i')
    hdf5.create_dataset("deaths", (num_images,), dtype='i')
    hdf5.create_dataset("img", shape)
    for i in range(num_images):
        img, score, admit, intub, death = dataset[i]
        if score > 1:
            hdf5["scores"][i] = int(1)
        else:
            hdf5["scores"][i] = int(0)
        hdf5["admits"][i] = admit
        hdf5["intubs"][i] = intub
        hdf5["deaths"][i] = death
        hdf5["img"][i, ...] = img
        if (i + 1) % 10 == 0:
            print(f"{hdf5_file_name}: {i + 1}/{num_images} images completed")
        elif i + 1 == num_images:
            print(f"{hdf5_file_name}: {i + 1}/{num_images} images completed")
