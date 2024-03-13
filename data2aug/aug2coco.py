import glob
import os
import re
import dataclasses
import json
from datetime import datetime

import cv2
import pytz
from PIL import Image


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

"""## Augmentation

### Imports
"""
import random

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa, BoundingBox as bb, Keypoint as kp
from imgaug import BoundingBoxesOnImage as bbi, KeypointsOnImage as kpi
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from datetime import datetime
import pytz
from imgaug import Polygon as poly

from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

# stupid little numpy deprecation
np.bool = np.bool_

"""### dict <-> location transformation"""

def dict_to_bb(d_boxes):
    b_boxes = []

    for d_box in d_boxes:
        b_boxes.append(
            bb(x1=d_box["x1"], y1=d_box["y1"], x2=d_box["x2"], y2=d_box["y2"])
        )

    return b_boxes


def bb_to_dict(b_boxes):
    d_boxes = []
    for b_box in b_boxes.bounding_boxes:
        diction = {
            "x1": float(b_box.x1),
            "y1": float(b_box.y1),
            "x2": float(b_box.x2),
            "y2": float(b_box.y2)
        }
        d_boxes.append(diction)
    return d_boxes


def dict_to_kp(d_boxes):
    keypoints = []

    for d_box in d_boxes:
        x1 = d_box["x1"]
        y1 = d_box["y1"]
        x2 = d_box["x2"]
        y2 = d_box["y2"]
        keypoints.extend(
            [kp(x1, y1),
             kp(x1, y2),
             kp(x2, y2),
             kp(x2, y1)]
        )

    return keypoints


def kp_to_dict(keypoints, width, height):
    d_boxes = []

    i = 0
    while i < len(keypoints.keypoints):
        k0 = keypoints.keypoints[i]
        k1 = keypoints.keypoints[i + 1]
        k2 = keypoints.keypoints[i + 2]
        k3 = keypoints.keypoints[i + 3]
        diction = {
            "x0": float(k0.x),
            "y0": float(k0.y),
            "x1": float(k1.x),
            "y1": float(k1.y),
            "x2": float(k2.x),
            "y2": float(k2.y),
            "x3": float(k3.x),
            "y3": float(k3.y)
        }
        d_boxes.append(diction)
        i += 4

    return d_boxes


def dict_to_mask(d_boxes, width, height):
    segmap = np.zeros((height, width), dtype=bool)

    for i in range(len(d_boxes)):
        d_box = d_boxes[i]

        x1 = d_box["x1"]
        y1 = d_box["y1"]
        x2 = d_box["x2"]
        y2 = d_box["y2"]
        segmap[y1:y2, x1:x2] = i + 1

    return segmap

"""### Coco segmentation"""

SIDE_POINTS_NUM = 5


def dict_to_gen_kp(d_boxes):
    def delete_extra(arr):
        return arr[:SIDE_POINTS_NUM]

    keypoints = []

    for d_box in d_boxes:
        x1 = d_box["x1"]
        y1 = d_box["y1"]
        x2 = d_box["x2"]
        y2 = d_box["y2"]

        interv_x = (x2 - x1) / (SIDE_POINTS_NUM + 1)
        interv_y = (y2 - y1) / (SIDE_POINTS_NUM + 1)

        x_list = np.arange(x1 + interv_x, x2, interv_x).tolist()
        x_list = delete_extra(x_list)

        y_list = np.arange(y1 + interv_y, y2, interv_y).tolist()
        y_list = delete_extra(y_list)
        keypoints.append(kp(x1, y1))  # (p0)

        for x in x_list:
            keypoints.append(kp(x, y1))  # (p0) .. (p1)

        keypoints.append(kp(x2, y1))  # (p1)

        for y in y_list:
            keypoints.append(kp(x2, y))  # (p1) .. (p2)

        keypoints.append(kp(x2, y2))  # (p2)

        for x in x_list:
            keypoints.append(kp(x, y2))  # (p2) .. (p3)

        keypoints.append(kp(x1, y2))  # (p3)

        for y in y_list:
            keypoints.append(kp(x1, y))  # (p3) .. (p0)

        # keypoints.append(kp(x1, y1))  # (p0) for closure

        # 4 + 4 * SIDE_POINTS_NUM point in total
    return keypoints


def divide_points(keypoints):
    total_points = 4 + 4 * SIDE_POINTS_NUM

    block_num = len(keypoints) // total_points
    print(len(keypoints))
    block_points = []
    for keypoint in keypoints:
        block_points.extend([keypoint.x, keypoint.y])

    segmentations = np.reshape(
        np.array(block_points), (block_num, total_points * 2)
    ).tolist()

    return segmentations


def filter_n_cast_points(points, width, height):
    def check_out(coord, crit):
        return coord < 0 or coord > crit

    def f(coord):
        return float(coord)

    filtered_points = []
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i + 1]
        if (not check_out(x, width)) and (not check_out(y, height)):
            filtered_points.extend([f(x), f(y)])

        elif check_out(x, width) and not check_out(y, height):
            if x > width:
                filtered_points.extend([f(width), f(y)])
            else:
                filtered_points.extend([0, f(y)])

        elif (not check_out(x, width)) and check_out(y, height):
            if y > height:
                filtered_points.extend([f(x), f(height)])
            else:
                filtered_points.extend([f(x), 0])

    return filtered_points


def get_segmentations(keypoints, width, height):
    filtered_segmentation = []
    for points in divide_points(keypoints):
        filtered_points = filter_n_cast_points(points, width, height)
        filtered_points = (np.reshape(filtered_points, (1, len(filtered_points)))
                           .tolist())

        filtered_segmentation.append(filtered_points)

    return filtered_segmentation

def bb_to_coco_list(b_boxes):
    list_coco_boxes = []
    b_boxes = b_boxes.remove_out_of_image().clip_out_of_image()
    for b_box in b_boxes.bounding_boxes:
        x_min, x_max = min(b_box.x1, b_box.x2), max(b_box.x1, b_box.x2)
        width = x_max - x_min

        y_min, y_max = min(b_box.y1, b_box.y2), max(b_box.y1, b_box.y2)
        height = y_max - y_min

        coco_box = [float(x_min), float(y_min), float(width), float(height)]
        list_coco_boxes.append(coco_box)

    return list_coco_boxes

def segmentations_to_xy(segmentations):
    xy_list = []

    for segmentation in segmentations:
        points = segmentation[0]
        xy = []

        for i in range(0, len(points), 2):
            x = points[i]
            y = points[i + 1]
            xy.append([x, y])

        xy_list.append(xy)
    return xy_list


def calculate_areas(xy_list):
    area_list = []

    for xy in xy_list:
        polygon = poly(xy)

        area = polygon.area
        area_list.append(area)

    return area_list

"""### Sequence build

Complex(standart) sequence:
"""

def build_heavy_sequence():
    seq = iaa.Sequential(
        [
            iaa.SomeOf((1, 5), [
                # arithmetic pixel transformation
                iaa.OneOf([

                    # noise
                    iaa.OneOf([
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),

                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                        iaa.AdditivePoissonNoise(
                            lam=(0.0, 15), per_channel=0.5
                        )
                    ]),

                    # pixel dropout(simple+large rectangles)
                    iaa.OneOf([
                        iaa.Dropout(
                            per_channel=0.5, p=(0.0, 0.1)
                        ),
                        iaa.CoarseDropout(
                            p=(0.0, 0.05),
                            per_channel=True,
                            size_percent=(0.02, 0.09)
                        )
                    ]),
                ]),

                # DefocusBlur: the picture was taken with defocused camera
                # MotionBlur: the picture was taken in motion
                iaa.OneOf([
                    iaa.imgcorruptlike.DefocusBlur(severity=1),
                    iaa.imgcorruptlike.MotionBlur(severity=1)
                ]),

                # image quality worsening
                iaa.OneOf([
                    iaa.imgcorruptlike.JpegCompression(severity=2),
                    iaa.imgcorruptlike.Pixelate(severity=2)
                ]),

                # contrast
                iaa.OneOf([
                    iaa.AllChannelsCLAHE(clip_limit=(1, 10)),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                ]),

                # gradient brightness+color change
                iaa.OneOf([
                    iaa.BlendAlphaHorizontalLinearGradient(
                        iaa.SomeOf((1, 2), [
                            iaa.Multiply((0.5, 1.5), per_channel=False),
                            iaa.ChangeColorTemperature((2000, 20000))
                        ]),
                        min_value=0.2, max_value=0.8),
                ]),

                # image resize
                iaa.SomeOf((1, 2), [
                    iaa.ScaleX(scale=(0.5, 1.5)),
                    iaa.ScaleY(scale=(0.5, 1.5))
                ]),

                iaa.OneOf([
                    iaa.PerspectiveTransform(scale=(0.01, 0.10), keep_size=False),
                    iaa.PerspectiveTransform(scale=(0.01, 0.10), keep_size=True),
                    iaa.Rotate((-30, 30)),
                    iaa.Rot90((1, 3))
                ]),

                iaa.OneOf([
                    iaa.Invert(0.2, per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                ])
            ])

        ], random_order=True
    )

    return seq

"""Custom sequence(don't forget to change the param of trans_image_boxes call):"""

def build_custom_sequence():
    seq = iaa.Rotate((-15, 15))

    return seq

"""### Data transformation"""

def trans_image_boxes(image, d_boxes, k, k2, k3, seq):
    kps = kpi(dict_to_kp(d_boxes), shape=image.shape)
    bbs = bbi(dict_to_bb(d_boxes), shape=image.shape)

    im, points, boxes = seq(image=image, keypoints=kps, bounding_boxes=bbs)
    h, w, _ = im.shape

    save_image(im, k + '.png')
    j_points = kp_to_dict(points, w, h)
    save_json(j_points, k2 + 'point.json')
    j_boxes = bb_to_dict(boxes)
    save_json(j_boxes, k3 + 'boxes.json')

def transform_data(image_files, box_files, output_dir, output_dir2, output_dir3):
    n = len(image_files)

    for i in range(n):
        data = get_image_and_boxes(image_files[i], box_files[i])

        boxes = data["boxes"]
        image = data["image"]

        time = datetime.now(pytz.timezone('Europe/Kiev'))
        name = time.strftime("%d-%m-%Y_%H-%M-%S_%f")

        trans_image_boxes(image, boxes, output_dir + name, output_dir2 + name,
                           output_dir3 + name,
                           build_heavy_sequence())

"""### Coco data transformation"""

class Coco:
    def __init__(self, image, bbox, segmentation, area, pil):
        self.image = image
        self.category = "Table"
        self.bbox = bbox
        self.segmentation = segmentation
        self.area = area
        self.PIL = pil

    def to_coco_dict(self):
        return {
            "image": self.image,
            "category": self.category,
            "bbox": self.bbox,
            "segmentation": self.segmentation,
            "area": self.area,
            "PIL": self.PIL
        }

def add_coco_anotations(map, h, w, coco_list, im_name, max_block_num):
    def adjust_negative(list):
        for i in range(len(list)):
            if list[i] < 0:
                list[i] = 0

    contours = measure.find_contours(map.arr.reshape(h, w), 0.5, positive_orientation='low')
    if max_block_num < len(contours):
        return False

    for contour in contours:

        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        coordinates = np.array(poly.exterior.coords)
        segmentation = coordinates.ravel().tolist()

        adjust_negative(segmentation)
        segmentations = [segmentation]

        x, y, max_x, max_y = poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = [x, y, width, height]
        adjust_negative(bbox)
        area = poly.area

        if area <= 0:
            return False

        coco = Coco(im_name + '.png', bbox, segmentations, area)
        coco_list.append(coco.to_coco_dict())
    return True

def coco_trans_image_boxes(image, d_boxes, im_name, output_dir, seq, coco_list):
    kps = kpi(dict_to_gen_kp(d_boxes), shape=image.shape)
    bbs = bbi(dict_to_bb(d_boxes), shape=image.shape)

    im, points, boxes = seq(image=image, keypoints=kps, bounding_boxes=bbs)
    h, w, _ = im.shape

    coco_segmentations = get_segmentations(points, w, h)
    coco_boxes = bb_to_coco_list(boxes)
    areas = calculate_areas(segmentations_to_xy(coco_segmentations))

    n = len(coco_boxes)

    for i in range(n):
        coco = Coco(im_name + '.png', coco_boxes[i], coco_segmentations[i], areas[i])
        coco_list.append(coco.to_coco_dict())

    save_image(im, output_dir + im_name + '.png')


def coco_trans_image_boxes2(image, d_boxes, im_name, output_dir, seq, coco_list):
    h, w, _ = image.shape

    mask = dict_to_mask(d_boxes, w, h)
    max_block_num = mask.max()
    seg = SegmentationMapsOnImage(mask, shape=image.shape)

    im, aug_mask = seq(image=image, segmentation_maps=seg)
    h, w, _ = im.shape

    if add_coco_anotations(aug_mask, h, w, coco_list, im_name, max_block_num):
        save_image(im, output_dir + im_name + '.png')

def coco_transform_data(image_files, box_files, output_dir, rep_num):
    n = len(image_files)
    coco_list = []

    for _ in range(rep_num):
        for i in range(n):
            data = get_image_and_boxes(image_files[i], box_files[i])

            boxes = data["boxes"]
            image = data["image"]

            time = datetime.now(pytz.timezone('Europe/Kiev'))
            name = time.strftime("%d-%m-%Y_%H-%M-%S_%f")

            coco_trans_image_boxes2(image, boxes, name, output_dir,
                                    build_heavy_sequence(), coco_list)

    save_json(coco_list, output_dir + 'coco.json')


'''---------------------------'''
def vika_add_coco_anotations(image, map, h, w, im_name, max_block_num):
    coco_list = []

    def adjust_negative(list):
        for i in range(len(list)):
            if list[i] < 0:
                list[i] = 0

    contours = measure.find_contours(map.arr.reshape(h, w), 0.5, positive_orientation='low')
    if max_block_num < len(contours):
        return False

    for contour in contours:

        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        coordinates = np.array(poly.exterior.coords)
        segmentation = coordinates.ravel().tolist()

        adjust_negative(segmentation)
        segmentations = [segmentation]

        x, y, max_x, max_y = poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = [x, y, width, height]
        adjust_negative(bbox)
        area = poly.area

        if area <= 0:
            return False

        #img = Image.fromarray(image)

        coco = Coco(im_name[:-4] + '.png', bbox, segmentations, area, image)
        coco_list.append(coco.to_coco_dict())
    return coco_list

def aug_trans_image_boxes(image, d_boxes, im_name, seq):
    h, w, _ = image.shape

    mask = dict_to_mask(d_boxes, w, h)
    max_block_num = mask.max()
    seg = SegmentationMapsOnImage(mask, shape=image.shape)

    im, aug_mask = seq(image=image, segmentation_maps=seg)
    h, w, _ = im.shape

    return vika_add_coco_anotations(im, aug_mask, h, w, im_name, max_block_num)


def gen2aug(data):
    boxes = data["boxes"]

    open_cv_image = data["image"]
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    return aug_trans_image_boxes(open_cv_image, boxes, data['image_name'], build_heavy_sequence())


'''
dataset_path = '/content/drive/MyDrive/img2text/input_data/images/'
bounding_boxes_path = '/content/drive/MyDrive/img2text/input_data/table_locations/'
augmented_dir = '/content/drive/MyDrive/img2text/aug2coco/result2/'

transformed_dir = augmented_dir + 'aug'
converted_dir = augmented_dir + 'convert'

os.makedirs(augmented_dir, exist_ok=True)
os.makedirs(transformed_dir, exist_ok=True)
os.makedirs(converted_dir, exist_ok=True)

"""### Get files"""

image_files = get_source_img_files(dataset_path)

boxes = get_source_box_files(bounding_boxes_path)

"""### Transform data"""

coco_transform_data(image_files, boxes, transformed_dir, 1)
'''
