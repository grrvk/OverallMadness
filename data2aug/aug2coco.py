from data2aug.coco_transformation import trans_image_keypoints
from data2aug.sequence_build import build_heavy_sequence


def gen2aug(data, aug_times):
    coco_list = []
    for i in range(aug_times):
        boxes = data["boxes"]

        open_cv_image = data["image"]
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        img_name = data['image_name'][:-4] + '_' + str(i) + '.png'
        trans_image_keypoints(open_cv_image, boxes, img_name, build_heavy_sequence(), coco_list)

    return coco_list
