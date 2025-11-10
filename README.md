# CSSE416-breast-image-segmentation  
  
## envrironment requirement 
see details in requirements.txt  
  
## Web_stuff folder  
This folder contains the web python programme running on local host. This code originally comes from Lifei Liu, and was optimized by Chengyang Ye. When using the web app, upload files from the subfolder testSet so the labels can be uploaded automatically. To make this happen, the path to test folder has to be entered in the dataset folder on the left.  
  
There is a hugging face version: https://huggingface.co/spaces/OgakuRina/CSSE416-Breast-image-Segmentation  
But this hugging face host still requires to download testset in this repo and upload from local  

## YOLO
This folder contains the training results and scripts for **breast mass detection and classification** using Yolo, reference link : https://github.com/monajemi-arman/breast_cancer_detection

## V-Unet  
This is the main folder where training on image segmentation happens. This code mainly uses package segmentation_models.pytorch, link as follows:   
https://github.com/qubvel-org/segmentation_models.pytorch.git   
This contains the following structure: Unet, Unet++ and DeepLabV3+  
and the following pretained network: VGG16, ResNet50 and Xception
  
#### models  
this folder contains all the models trained in this project. For how files are named, the format is 'pretrainedNets_structrue_aa/ba_de/en'. 'aa' stands for after augmentation, 'ba' stands for before augmentation, 'de' stands for only unfreezing decoder and 'en' stands for unfreezing part of the encoder.  
  
#### tensorflow  
This folder is just some failure trials on using tensorflow package. Does not work due to environment setup.  
  
#### pytorch  
divide into aa and ba, and under them divide into de and en. Every Jupyter contains a training block and visualization block. Under ba there is a from scratch folder which does not use any pre-trained networks in the encoder. The reults are really jokes due to a small dataset.  
  
#### Visualization  
A code runs for visualizing all the models instead of going to every single juypter notebook to see the reults.  

## inbreast_extract_and_preprocess
This file preprocessed the dataset, turn labels into mask format, augmentated both images and masks, and split train/val/test into ratio 8/1/1

## pectoral muscles labeling reference link
https://github.com/Parvaneh-Aliniya/pectoral_muscle_groundtruth_segmentation


