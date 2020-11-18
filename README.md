# DL Algorithm for Prediction of Clinical Outcomes of COVID-19 Patients
This repository contains the source code for Kwon et al. "Deep Learning Algorithm Predicts Clinical Outcomes of COVID-19 Patients Based on Initial Chest Radiographs from the Emergency Department" submitted to *Radiology: Artificial Intelligence*. This algorithm was trained and tested on 499 total radiographs that were evaluated by fellowship trained, board certified radiologists. 

## Usage

### Patient selection
Patients with any CXR and routine laboratory tests from the initial emergency department encounter may be used. 

### Architecture
The main algorithm used inthis paper is the DenseNet-121 pre-trained on ImageNet, similar to the algorithm utilized in the [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/) paper. The train / validation / test split is summarized by the figure below.


### Prerequisites

* python3
* PyTorch (torch)
* torchvision
* HDF5 (h5py)
* numpy
* tqdm
* matplotlib
* scikit-learn (sklearn)


## Getting Started

### Create HDF5 Dataset

We used HDF5 datasets to create and save padded images such that the training does not require pre-processing each time. As noted above, all radiographs were center cropped. Final image size could be specified. The HDF5 dataset contains the images (normalized and stored as Tensors), severity scores, 30-day admission status, 30-day intubation status, and 30-day mortality.

```
python create_hdf5.py --img_size=1024 --crop_size=1024
```


### Training

1. To run the VQ VAE training script using the default hyperparameters. Save path saves loss functions and AUROCs per epoch and the best models selected by the model that results in minimum binary cross entropy (BCE) loss in the validation set:

```
python train_densenet.py --data_path=[HDF5 TRAIN DATASET PATH] --save_path=[SAVE PATH]
```

We trained using both the severity scores and the 30-day admission status to predict severity scores, 30-day admission status, 30-day intubation status, and 30-day mortality.

2. To prepare experiments with EHR, we used the code from [Nvidia](https://gitlab.com/nvidia/sa/covid-fl/-/tree/master/fed_learn) that is available with the [Clara Software Development Kit (SDK)](https://developer.nvidia.com/clara)

### Testing

1. Use the `test_model.ipynb` Jupyter Notebook to:
* create AUROC (area under the receiver operating characteristic curve) plots on the test set
* create PR (precision recall) curves on the test set


2. Use the `heatmap.py` to create desired heatmap of a radiograph:

```
python heatmap.py --index=[index of radiograph]
```


### Results

1. Performance of prediction of intubation (AUC, 0.88) and death (AUC, 0.82) increased with incorporation of relevant clinical variables from electronic health records acquired exclusively from the emergency department encounter.

2. The model, despite training with only young patients aged 21 to 50, generalized to a pseudo-prospective test set that also contained older patients aged greater than 50.


## Contributors

* **Young Joon (Fred) Kwon MS** |[github](https://github.com/kwonfred)|[linkedin](https://www.linkedin.com/in/kwonfred/)| MD PhD Student; Icahn School of Medicine at Mount Sinai
* Eric K Oermann MD |[github](https://github.com/RespectableGlioma)|[linkedin](https://www.linkedin.com/in/eric-oermann-b829528/)| Instructor, Department of Neurosurgery; Director, AISINAI; Icahn School of Medicine at Mount Sinai
* Anthony B Costa PhD |[github](https://github.com/acoastalfog)|[linkedin](https://www.linkedin.com/in/anthony-costa-17005a64/)| Assistant Professor, Department of Neurosurgery; Director, Sinai BioDesign; Icahn School of Medicine at Mount Sinai


## License

This project is licensed under the APACHE License, version 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details


## Acknowledgments

* MSTP T32 NIH T32 GM007280
* RSNA Medical Student Research Grant
* Intel Software and Services Group Research Grant
* The BioMedical Engineering and Imaging Institute, Icahn School of Medicine at Mount Sinai
* Mount Sinai Covid Informatics Center, Icahn School of Medicine at Mount Sinai
