#!/bin/bash
pip install --upgrade pip
### DATA DOWNLOAD ###
pip install kaggle
export KAGGLE_USERNAME=aasmundbrekke
export KAGGLE_KEY=2e4cb56a1863ae4d984e72004f63a6cc
# Uncomment the following lines if you do not have the dataset downloaded
kaggle competitions download -c rsna-pneumonia-detection-challenge
# unzipping takes a few minutes
unzip -q -o stage_1_test_images.zip -d stage_1_test_images
unzip -q -o stage_1_train_images.zip -d stage_1_train_images
unzip -q -o stage_1_train_labels.csv.zip

#####################
cd Mask_RCNN
pip install -r requirements.txt
python setup.py install
cd ..
pip install -r requirements.txt
#python run.py
python main.py stage_1_train_images/ stage_1_test_images/ stage_1_training_labels.csv
