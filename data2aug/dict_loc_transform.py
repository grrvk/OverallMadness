import numpy as np
from imgaug import BoundingBox as bb, Keypoint as kp


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


def kp_to_dict(keypoints, w, h):
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


def dict_to_gen_kp(d_boxes, side_points_num):
    def delete_extra(arr):
        return arr[:side_points_num]

    keypoints = []

    for d_box in d_boxes:
        x1 = d_box["x1"]
        y1 = d_box["y1"]
        x2 = d_box["x2"]
        y2 = d_box["y2"]

        interv_x = (x2 - x1) / (side_points_num + 1)
        interv_y = (y2 - y1) / (side_points_num + 1)

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

        # 4 + 4 * side_points_num point in total
    return keypoints
