import pydicom
import os
import cv2
import numpy as np
import glob
"""
Prepare the input data to match expected shape as darknet wants it. 

"""



def full_path(folder_name):
    """ Returns the full path to the given folder from the current working directory """
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.isdir(folder_path): # Make directory if it does not exist
	    os.mkdir(folder_path)
    return folder_path



DATA_DIR = "../data"

train_dir = full_path("images")
test_dir = full_path("test_images")
label_dir= full_path("labels")
meta_dir = full_path("metadata")

cfg_dir = full_path("cfg")
backup_dir = full_path("backup")






def write_dcm_to_jpg(dcm_img, dst_folder, filepath):
    """ Writes the dcm image as jpeg to the given filepath 
	The resulting jpg image will have 3 channels.
    """
    img = dcm_img.pixel_array
    img_channeled = np.stack([img]*3, -1)
    filepath = filepath.replace(".dcm", ".jpg")
    cv2.imwrite(os.path.join(dst_folder, filepath), img_channeled)


def convert_all_dcm_to_jpg(src_folder, dst_folder):
    """ Converts all dcm images in src_folder to jpgs which will be written to dst_folder """
    for file in os.listdir(src_folder):
	    if file.endswith(".dcm"):
	        src_path = os.path.join(src_folder, file)
	        dcm_img = pydicom.read_file(src_path)
	        write_dcm_to_jpg(dcm_img, dst_folder, file)
	    
def generate_yolo_labels(label_dir, filename, bbox_data = None):
    """
        Generates 
    """
    if bbox_data is None:
        return 
    
    if not filename.endswith(".txt"):
        filename += ".txt"
    IMG_W, IMG_H = (1024, 1024)
    label_filepath = os.path.join(label_dir, filename)
    
    x = bbox_data[0]/IMG_W
    y = bbox_data[1]/IMG_H
    w = bbox_data[2]/IMG_W
    h = bbox_data[3]/IMG_H
    # Since yolo wants x,y coords relative to the center
    x_center = x + w/2
    y_center = y + h/2
    
    with open(label_filepath, "a") as f:
        # Since we only want one class (pneumonia), call it 0
        f.write("0 {} {} {} {}".format(x_center, y_center, w, h))    
    
        

train_dcim_dir = os.path.join(DATA_DIR, "stage_1_train_images")
test_dcim_dir = os.path.join(DATA_DIR, "stage_1_test_images")



#convert_all_dcm_to_jpg(train_dcim_dir, train_dir)
#convert_all_dcm_to_jpg(test_dcim_dir, test_dir)

def write_yolo_annotations():
    annotations = pd.read_csv(os.path.join(DATA_DIR, "stage_1_train_labels.csv"))
    for row in annotations.values[1:]: # Skip header row
        # patientId, x, y, width, height, target
        if row[5] == 0: # Skip patients that have no lung densities
            continue 
        bbox_data = row[1:4]
        patientId = row[0]
        generate_yolo_labels(label_dir, patientId, bbox_data)


write_yolo_annotations()
