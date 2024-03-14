import glob
import os
import re
import json
from datetime import datetime

import cv2
import pytz


def get_source_img_files(path):
    """Function reading all the images with png/jpg/jpeg extension from the base dataset folder"""
    image_files = [file for file in glob.glob(path + '*')
                   if re.search(r'.+\.(png|jpg|jpeg)', file)]

    return image_files


def rename_source_files(image_files, f_structure):
    """Function renaming all the image files from the base dataset folder to match cv2 standards"""
    for image in image_files:
        time = datetime.now(pytz.timezone('Europe/Kiev'))
        name = time.strftime("%d-%m-%Y_%H-%M-%S_%f")
        os.rename(str(image), f_structure.dataset_path + name)


def get_source_images(image_files):
    """Function transforming image files into numpy arrays"""
    source_images = []

    for image in image_files:
        img = cv2.imread(str(image))
        source_images.append(img)

    return source_images


def get_source_box_files(path):
    """Function reading all the images with png/jpg/jpeg extension from the base dataset folder"""
    box_files = [file for file in glob.glob(path + '*')
                 if re.search(r'.+\.(json)', file)]

    return box_files


def get_box(box):
    """Function transforming image files into numpy arrays"""
    with open(box, 'r') as openfile:
        bounding_boxes = json.load(openfile)

    return bounding_boxes


def get_image(image):
    img = cv2.imread(str(image))
    return img


def get_image_and_boxes(image, box):
    img = cv2.imread(str(image))

    with open(box, 'r') as openfile:
        bounding_boxes = json.load(openfile)

    return {"boxes": bounding_boxes,
            "image": img}


def save_image(image, f_name):
    cv2.imwrite(f_name, image)


def save_json(location_dict, f_name):
    location_json = json.dumps(location_dict, indent=4)

    with open(f_name, "w") as outfile:
        outfile.write(location_json)


def save_json_object(location_dict, f_name):
    location_json = json.dumps(location_dict.toJson(), indent=4)

    with open(f_name, "w") as outfile:
        outfile.write(location_json)
