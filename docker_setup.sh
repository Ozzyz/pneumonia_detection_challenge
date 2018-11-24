#!/bin/bash
pip install --upgrade pip
### DATA DOWNLOAD ###
#pip install kaggle
# Uncomment the following lines if you do not have the dataset downloaded
#kaggle competitions download -c rsna-pneumonia-detection-challenge
# unzipping takes a few minutes
#unzip -q -o stage_1_test_images.zip -d stage_1_test_images
#unzip -q -o stage_1_train_images.zip -d stage_1_train_images
#unzip -q -o stage_1_train_labels.csv.zip

#####################
cd Mask_RCNN
pip install -r requirements.txt
python setup.py install
cd ..
pip install -r requirements.txt
#python run.py
#python main.py stage_1_train_images/ stage_1_test_images/ stage_1_training_labels.csv
