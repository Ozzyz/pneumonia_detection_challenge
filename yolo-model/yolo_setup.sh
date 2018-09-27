#!/bin/bash
### System updates ###
apt-get update
apt-get install -y vim


pip install --upgrade pip
### DATA DOWNLOAD ###
pip install kaggle
# Uncomment the following lines if you do not have the dataset downloaded
#kaggle competitions download -c rsna-pneumonia-detection-challenge
# unzipping takes a few minutes
#unzip -q -o stage_1_test_images.zip -d stage_1_test_images
#unzip -q -o stage_1_train_images.zip -d stage_1_train_images
#unzip -q -o stage_1_train_labels.csv.zip

#####################
