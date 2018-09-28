import os
import sys
import numpy as np
import pydicom
from tqdm import tqdm
import glob
from config import DetectorConfig, InferenceConfig
import cv2
import matplotlib.pyplot as plt

visualize = False

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

test_dicom_dir = os.path.join(ROOT_DIR, 'data/stage_1_test_images')

# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024
config = DetectorConfig()
print(config.display())


def get_dicom_fps(dicom_dir):
	print(dicom_dir)
	print(dicom_dir + '/' + '*.dcm')
	dicom_fps = glob.glob(dicom_dir + '/' + '*.dcm')
	return list(set(dicom_fps))


def parse_dataset(dicom_dir, anns):
	image_fps = get_dicom_fps(dicom_dir)
	image_annotations = {fp: [] for fp in image_fps}
	print(len(image_annotations))
	for index, row in anns.iterrows():
		fp = os.path.join(dicom_dir, row['patientId'] + '.dcm')
		image_annotations[fp].append(row)
	return image_fps, image_annotations


def run():
	print(visualize)
	# select trained model
	dir_names = next(os.walk(MODEL_DIR))[1]
	print(MODEL_DIR)
	key = config.NAME.lower()
	dir_names = filter(lambda f: f.startswith(key), dir_names)
	dir_names = sorted(dir_names)
	print(dir_names)
	if not dir_names:
		import errno
		raise FileNotFoundError(
			errno.ENOENT,
			"Could not find model directory under {}".format(MODEL_DIR))

	fps = []
	# Pick last directory
	for d in dir_names:
		dir_name = os.path.join(MODEL_DIR, d)
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
	# resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
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
			assert (len(r['rois']) == len(r['class_ids']) == len(r['scores']))
			if len(r['rois']) == 0:
				print("Predicted empty")
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
						bboxes_str = "{} {} {} {}".format(x1 * resize_factor, y1 * resize_factor,
														width * resize_factor, height * resize_factor)
						out_str += bboxes_str

						if visualize:
							cv2.rectangle(image, (x1, y1), (width, height), (0, 255, 0), 2)

				if visualize:
					plt.imshow(image)
					#plt.show()
					plt.pause(0.01)

			file.write(out_str + "\n")


if __name__ == "__main__":
	for arg in sys.argv[1:]:
		if arg.lower() == 'visualize':
			visualize = True
	print("ROOT_DIR: ", ROOT_DIR)
	run()
