import datetime
from dataclasses import dataclass


@dataclass
class InfoClass:
    year: int = datetime.date.today().year
    version: str = "0.1.0"
    description: str = ""
    contributor: str = ""
    url: str = ""
    date_created: datetime = datetime.date.today()

    def __init__(self, version: str = "0.1.0", description: str = "",
                 contributor: str = "", url: str = ""):
        self.year = datetime.date.today().year
        self.version = version
        self.description = description
        self.contributor = contributor
        self.url = url
        self.date_created = datetime.date.today()


@dataclass
class CategoryClass:
    name: str
    id: int
    supercategory: int | None

    def __init__(self, name: str, id: int, supercategory: int | None):
        self.id = id
        self.name = name
        self.supercategory = supercategory


@dataclass
class ImageClass:
    id: int
    width: int
    height: int
    file_name: str
    date_captured: datetime = datetime.date.today()

    def __init__(self, id: int, width: int, height: int, file_name: str):
        self.id = id
        self.width = width
        self.height = height
        self.file_name = file_name
        self.date_captured  = datetime.date.today()


@dataclass
class AnnotationClass:
    id: int
    image_id: int
    category_id: int
    segmentation: list
    area: float
    bbox: list
    iscrowd: int

    def __init__(self, id: int, image_id: int, category_id: int, segmentation: list, area: float,
                 bbox: list, iscrowd: int):
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.area = area
        self.bbox = bbox
        self.iscrowd = iscrowd


class JsonFileClass:
    info: dict
    categories: list
    images: list
    annotations: list

    def __init__(self, info: dict, categories: list, images: list, annotations: list):
        self.info = info
        self.categories = categories
        self.images = images
        self.annotations = annotations


class PrevDataInfoHolder:
    CATEGORIES_NEW: list = []
    CATEGORIES_TOTAL: list = []
    INFO: dict

    IMAGE_INDEX: int = 0
    ANNOTATION_INDEX: int = 0

    def __init__(self):
        self.IMAGE_INDEX = 0
        self.ANNOTATION_INDEX = 0
        self.CATEGORIES = []


    def set_changes(self, imI: int = 0, anI: int = 0):
        self.IMAGE_INDEX = imI
        self.ANNOTATION_INDEX = anI




