import numpy as np
from shapely.geometry import Polygon
from skimage import measure

from data2aug.coco import Coco


def divide_points(keypoints, side_points_num):
    total_points = 4 + 4 * side_points_num
    block_num = len(keypoints) // total_points

    block_points = []
    for keypoint in keypoints:
        block_points.extend([keypoint.x, keypoint.y])

    segmentations = np.reshape(
        np.array(block_points), (block_num, total_points * 2)
    ).tolist()

    return segmentations


def intersect_poly(points, w, h):
    im_xy = [[0, 0], [w, 0], [w, h], [0, h]]
    image = Polygon(im_xy)

    obj_xy = []
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i + 1]

        obj_xy.append([x, y])

    obj = Polygon(obj_xy).simplify(1.0, preserve_topology=False)

    if image.contains(obj):
        coords = np.array(obj.exterior.coords)
        return [coords.ravel().tolist()], obj.area

    inters = image.intersection(obj).simplify(1.0, preserve_topology=False)
    coords = np.array(inters.exterior.coords)

    return [coords.ravel().tolist()], inters.area


def get_segmentations(keypoints, width, height, side_points_num):
    inter_segmentation = []

    segmentations = divide_points(keypoints, side_points_num)
    areas = []

    for segment in segmentations:
        inter_segment, area = intersect_poly(segment, width, height)

        if not len(inter_segment) == 0:
            areas.append(area)
            inter_segmentation.append(inter_segment)

    return inter_segmentation, areas


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


def mask_to_annotation(image, map, h, w, im_name, max_block_num):
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

        # img = Image.fromarray(image)

        coco = Coco(im_name[:-4] + '.png', bbox, segmentations, area, image)
        coco_list.append(coco.to_coco_dict())
    return True
