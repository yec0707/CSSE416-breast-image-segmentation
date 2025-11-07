# CSSE416-breast-image-segmentation  

## Web_stuff folder  
This folder contains the web python programme. This code originally comes from Lifei Liu, and was optimized by Chengyang Ye. When using the web app, upload files from the subfolder testSet so the labels can be uploaded automatically. To make this happen, the path to test folder has to be entered in the dataset folder on the left.  
  
## V-Unet  
This is the main folder where training on image segmentation happens. This code mainly uses package segmentation_models.pytorch, link as follows:   
https://github.com/qubvel-org/segmentation_models.pytorch.git   
  
#### models  
this folder contains all the models trained in this project. For how files are named, the format is 'pretrainedNets_structrue_aa/ba_de/en'. 'aa' stands for after augmentation, 'ba' stands for before augmentation, 'de' stands for only unfreezing decoder and 'en' stands for unfreezing part of the encoder.  
  
#### tensorflow  
This folder is just some failure trials on using tensorflow package. Does not work due to environment setup.  
  
#### pytorch  
divide into aa and ba, and under them divide into de and en. Every Jupyter contains a training block and visualization block. Under ba there is a from scratch folder which does not use any pre-trained networks in the encoder. The reults are really jokes due to a small dataset.  
  
#### Visualization  
A code runs for visualizing all the models instead of going to every single juypter notebook to see the reults.  
