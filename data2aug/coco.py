from dataclasses import dataclass


@dataclass
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
