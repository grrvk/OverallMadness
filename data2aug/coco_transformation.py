from datetime import datetime
import pytz
from imgaug import BoundingBoxesOnImage, KeypointsOnImage, SegmentationMapsOnImage

from data2aug.dict_loc_transform import dict_to_gen_kp, dict_to_bb, dict_to_mask
from data2aug.coco_segmentation import get_segmentations, bb_to_coco_list, mask_to_annotation
from data2aug.coco import Coco
from data2aug.sequence_build import build_heavy_sequence
from data2aug.functions import save_image, get_image_and_boxes, save_json


def trans_image_keypoints(image, d_boxes, im_name, seq, coco_list):
    side_points_num = 5
    kps = KeypointsOnImage(dict_to_gen_kp(d_boxes, side_points_num), shape=image.shape)

    bbs = BoundingBoxesOnImage(dict_to_bb(d_boxes), shape=image.shape)

    aug_image, points, boxes = seq(image=image, keypoints=kps, bounding_boxes=bbs)
    # print(im_name, points, boxes)
    h, w, _ = aug_image.shape

    coco_segmentations, areas = get_segmentations(points, w, h, side_points_num)
    coco_boxes = bb_to_coco_list(boxes)

    for i in range(len(coco_boxes)):
        coco = Coco(im_name, coco_boxes[i], coco_segmentations[i], areas[i], aug_image)
        coco_list.append(coco.to_coco_dict())


def trans_image_mask(image, d_boxes, im_name, output_dir, seq, coco_list):
    h, w, _ = image.shape

    mask = dict_to_mask(d_boxes, w, h)
    max_block_num = mask.max()
    seg = SegmentationMapsOnImage(mask, shape=image.shape)

    im, aug_mask = seq(image=image, segmentation_maps=seg)
    h, w, _ = im.shape

    if mask_to_annotation(aug_mask, h, w, coco_list, im_name, max_block_num):
        save_image(im, output_dir + im_name + '.png')


def aug_trans_image_boxes(image, d_boxes, im_name, seq):
    h, w, _ = image.shape

    mask = dict_to_mask(d_boxes, w, h)
    max_block_num = mask.max()
    seg = SegmentationMapsOnImage(mask, shape=image.shape)

    im, aug_mask = seq(image=image, segmentation_maps=seg)
    h, w, _ = im.shape

    return  # vika_add_coco_anotations(im, aug_mask, h, w, im_name, max_block_num)


def transform_data(image_files, box_files, output_dir, rep_num):
    n = len(image_files)
    coco_list = []

    for _ in range(rep_num):
        for i in range(n):
            data = get_image_and_boxes(image_files[i], box_files[i])

            boxes = data["boxes"]
            image = data["image"]

            time = datetime.now(pytz.timezone('Europe/Kiev'))
            name = time.strftime("%d-%m-%Y_%H-%M-%S_%f")

            trans_image_keypoints(image, boxes, name, output_dir,
                                  build_heavy_sequence(), coco_list)

    save_json(coco_list, output_dir + 'coco.json')
