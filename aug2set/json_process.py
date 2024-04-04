import json
import os
import numpy as np
import cv2
from PIL import Image
from pandas import DataFrame
from aug2set.schemes import InfoClass, CategoryClass, ImageClass, AnnotationClass, JsonFileClass
from aug2set.data_process import _findJsonFolder


def _setInfo():
    """
    :return: dictionary of info class for Coco
    """
    return InfoClass().__dict__


def _setCategories(cats, index_increment):
    """
    :param cats: list of category names
    :param index_increment: index from which to set id (necessary if uploading)
    :return: list of dictionaries of categories for Coco
    """
    categories = []
    for i, cat in enumerate(cats):
        if type(cat) is str:
            category = CategoryClass(cat, i + 1 + index_increment, None)
        else:
            raise Exception('Category type is not str ')
        categories.append(category.__dict__)
    return categories


def set_constant(upload: bool, dataset_path: str, df: DataFrame):
    """
    :param upload: creating new or uploading to dataset
    :param dataset_path: creating/updating dataset path
    :param df: dataframe of new data
    :return: info class dict, categories class list, list of total category dict (new + uploaded if present)
    """
    json_files = _findJsonFolder(os.path.join(dataset_path, 'annotations'))

    if upload and len(json_files) != 0:
        with open(os.path.join(os.path.join(dataset_path, 'annotations'), json_files[0]), "r") as json_file:
            json_data = json.load(json_file)
            categories_json = list(category for category in json_data['categories'])

        category_names = [c['name'] for c in categories_json]
        new_categories = list(set(df[~df["category"].isin(category_names)]['category'].tolist()))
        categories = _setCategories(new_categories, len(category_names))
    else:
        categories_json = []
        categories = _setCategories(list(set(df["category"].tolist())), 0)
    return _setInfo(), categories, categories + categories_json


def _setImages(settings, df, index_increment, split_type):
    """
    :param settings: settings of paths and splits
    :param df: dataframe with new data
    :param index_increment: index from which to set id (necessary if uploading)
    :param split_type: train/val/test
    :return: list of dictionaries of images for Coco
    """
    images = []
    image_paths = list(set(df['image'].tolist()))
    for i, image_path in enumerate(image_paths):
        img = df[df['image'] == image_path]['PIL'].values[0]

        with open(f"{os.path.join(os.path.join(settings.DATASET_DIR, split_type), image_path)}", "wb") as f:
            f.write(cv2.imencode('.png', img)[1].tobytes())
        img = Image.fromarray(img)
        image = ImageClass(i + 1 + index_increment, img.width, img.height, image_path.split("/")[-1])
        images.append(image.__dict__)

    return images


def findObjectId(array, item_name, sorter):
    """
    :param array: array to search in
    :param item_name: value to search
    :param sorter: parameter name to search by
    :return: id of found item or None
    """
    item = next((item for item in array if item.get(sorter) == item_name), None)
    if not item:
        raise Exception(f"Cannot find item {item_name} from data passed in JSON in working directory")
    return item.get("id")


def _setAnnotations(df, categories, images, index_increment):
    """
    :param df: dataframe of new data
    :param categories: list of all category dicts
    :param images: list of all image dicts
    :param index_increment: index from which to set id (necessary if uploading)
    :return: list of dictionaries of annotations for Coco
    """
    annotations = []
    for index, row in df.iterrows():
        image_id = findObjectId(images, row['image'], "file_name")
        category_id = findObjectId(categories, row['category'], "name")
        segmentation = row['segmentation']
        area = row['area']
        bbox = row['bbox']
        annotation = AnnotationClass(
            index + 1 + index_increment, image_id, category_id, segmentation, area,
            bbox, 0
        )
        annotations.append(annotation.__dict__)
    return annotations


def _fillJson(settings, df, holder, split_type):
    """
    :param settings: settings of paths and splits
    :param df: dataframe of new data
    :param holder: class which contains previous/unchanging (info, categories) data and indexes
    :param split_type: train/val/test
    :return: dict of data for Coco json
    """
    images = _setImages(settings, df, holder.IMAGE_INDEX, split_type)
    annotations = _setAnnotations(df, holder.CATEGORIES_TOTAL, images, holder.ANNOTATION_INDEX)

    data = JsonFileClass(holder.INFO, holder.CATEGORIES_TOTAL, images, annotations)

    return data.__dict__












