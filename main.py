import os
import sys
import pydicom
import random
import math
import numpy as np
#import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import glob
from config import DetectorConfig, DetectorDataset, InferenceConfig

# enter your Kaggle credentionals here
os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY']=""

# Root directory of the project
ROOT_DIR = os.path.abspath('./')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

if not os.path.exists(ROOT_DIR):
	os.makedirs(ROOT_DIR)
os.chdir(ROOT_DIR)

# Import Mask RCNN
# To find local version of the library
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))
import mrcnn.model as modellib
from mrcnn.model import log

train_dicom_dir = os.path.join(ROOT_DIR, 'data', 'stage_1_train_images')
test_dicom_dir = os.path.join(ROOT_DIR, 'data', 'stage_1_test_images')

# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024
config = DetectorConfig()
print(config.display())

def get_dicom_fps(dicom_dir):
	print(dicom_dir)
	print(os.path.join(dicom_dir, '*.dcm'))
	dicom_fps = glob.glob(os.path.join(dicom_dir, '*.dcm'))
	return list(set(dicom_fps))


def parse_dataset(dicom_dir, anns):
	image_fps = get_dicom_fps(dicom_dir)
	image_annotations = {fp: [] for fp in image_fps}
	for index, row in anns.iterrows():
		fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
		image_annotations[fp].append(row)
	return image_fps, image_annotations


def run():
	# training dataset
	anns = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'stage_1_train_labels.csv'))
	print(anns.head(6))

	image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)

	#ds = pydicom.read_file(image_fps[0])  # read dicom image from filepath
	#image = ds.pixel_array  # get image array

	######################################################################
	# Modify this line to use more or fewer images for training/validation.
	# To use all images, do: image_fps_list = list(image_fps)
	image_fps_list = list(image_fps)
	#####################################################################
	print("Reading", len(image_fps_list), " image filepaths") 
	# split dataset into training vs. validation dataset
	# split ratio is set to 0.9 vs. 0.1 (train vs. validation, respectively)
	sorted(image_fps_list)
	random.seed(42)
	random.shuffle(image_fps_list)

	validation_split = 0.15
	split_index = int((1 - validation_split) * len(image_fps_list))

	image_fps_train = image_fps_list[:split_index]
	image_fps_val = image_fps_list[split_index:]
	print("Number of train images, number of validation images:")
	print(len(image_fps_train), len(image_fps_val))

	# prepare the training dataset
	dataset_train = DetectorDataset(
		image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
	dataset_train.prepare()
	
	# prepare the validation dataset
	dataset_val = DetectorDataset(
		image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
	dataset_val.prepare()
	print("Training with model dir:", MODEL_DIR)
	model = modellib.MaskRCNN(
		mode='training', config=config, model_dir=MODEL_DIR)

	# Image augmentation
	augmentation = iaa.SomeOf((0, 1), [
		iaa.Fliplr(0.5),
		iaa.Affine(
			scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
			translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
			rotate=(-25, 25),
			shear=(-8, 8)
		),
		iaa.Multiply((0.9, 1.1))
	])

	NUM_EPOCHS = 20

	# Train Mask-RCNN Model
	import warnings
	warnings.filterwarnings("ignore")
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=NUM_EPOCHS,
				layers='all',
				augmentation=augmentation)

	# select trained model
	dir_names = next(os.walk(model.model_dir))[1]
	print(model.model_dir)
	key = config.NAME.lower()
	dir_names = filter(lambda f: f.startswith(key), dir_names)
	dir_names = sorted(dir_names)
	print(dir_names)
	if not dir_names:
		import errno
		raise FileNotFoundError(
			errno.ENOENT,
			"Could not find model directory under {}".format(model.model_dir))

	fps = []
	# Pick last directory
	for d in dir_names:
		dir_name = os.path.join(model.model_dir, d)
		# Find the last checkpoint
		checkpoints = next(os.walk(dir_name))[2]
		checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
		checkpoints = sorted(checkpoints)
		if not checkpoints:
			print('No weight files in {}'.format(dir_name))
		else:
			checkpoint = os.path.join(dir_name, checkpoints[-1])
			fps.append(checkpoint)

	model_path = sorted(fps)[-1]
	print('Found model {}'.format(model_path))

	inference_config = InferenceConfig()

	# Recreate the model in inference mode
	model = modellib.MaskRCNN(mode='inference',
							  config=inference_config,
							  model_dir=MODEL_DIR)

	# Load trained weights (fill in path to trained weights here)
	assert model_path != "", "Provide path to trained weights"
	print("Loading weights from ", model_path)
	model.load_weights(model_path, by_name=True)
	"""
	# Show few example of ground truth vs. predictions on the validation dataset
	dataset = dataset_val
	fig = plt.figure(figsize=(10, 30))

	for i in range(4):
		image_id = random.choice(dataset.image_ids)

		original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
			modellib.load_image_gt(dataset_val, inference_config,
								   image_id, use_mini_mask=False)

		plt.subplot(6, 2, 2*i + 1)
		visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
									dataset.class_names,
									colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])

		plt.subplot(6, 2, 2*i + 2)
		results = model.detect([original_image])  # , verbose=1)
		r = results[0]
		visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
									dataset.class_names, r['scores'],
									colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
	"""
	# Get filenames of test dataset DICOM images
	test_image_fps = get_dicom_fps(test_dicom_dir)
	# Write predictions to file
	predict(model, test_image_fps, filepath="actual_submission_newest.csv")


def get_colors_for_class_ids(class_ids):
	colors = []
	for class_id in class_ids:
		if class_id == 1:
			colors.append((.941, .204, .204))
	return colors


def predict(model, image_fps, filepath='sample_submission.csv', min_conf=0.9):
	""" Makes predictions on test images, write out sample submission"""
	# assume square image
	#resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
	resize_factor = 1
	print("Predicting with resize-factor : ", resize_factor)
	with open(filepath, 'w') as file:
		for image_id in tqdm(image_fps):
			ds = pydicom.read_file(image_id)
			image = ds.pixel_array

			# If grayscale. Convert to RGB for consistency.
			if len(image.shape) != 3 or image.shape[2] != 3:
				image = np.stack((image,) * 3, -1)

			patient_id = os.path.splitext(os.path.basename(image_id))[0]

			results = model.detect([image])
			r = results[0]

			out_str = ""
			out_str += patient_id + ","
			assert(len(r['rois']) == len(r['class_ids']) == len(r['scores']))
			if len(r['rois']) == 0:
				pass
			else:
				num_instances = len(r['rois'])
				for i in range(num_instances):
					if r['scores'][i] > min_conf:
						out_str += ' '
						out_str += str(round(r['scores'][i], 2))
						out_str += ' '

						# x1, y1, width, height
						x1 = r['rois'][i][1]
						y1 = r['rois'][i][0]
						width = r['rois'][i][3] - x1
						height = r['rois'][i][2] - y1
						bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor,
														  width*resize_factor, height*resize_factor)
						out_str += bboxes_str

			file.write(out_str+"\n")


if __name__ == "__main__":
	print("ROOT_DIR", ROOT_DIR);
	run()
