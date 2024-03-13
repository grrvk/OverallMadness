"""
    This module contains classes that represent brochure and it`s components
"""


class Brochure:
    """
        This class contains parameters of generated brochure
    """
    def __init__(self, rows=0, columns=0, width=0, height=0, layout_matrix=None, text_parts=0,
                 left_margin=0, top_margin=0, header_height=0, image=None, draw=None):

        self.rows = rows
        self.columns = columns
        self.width = width
        self.height = height
        self.layout_matrix = layout_matrix
        self.text_parts = text_parts
        self.left_margin = left_margin
        self.top_margin = top_margin
        self.header_height = header_height
        self.image = image
        self.draw = draw

        self.t_cords = None

    def set_table_coords_arr(self):
        self.t_cords = []

    def add_coords(self, coords: dict):
        self.t_cords.append(coords)


class BrochureBlock:
    """
        This class contains parameters of brochure block
    """
    def __init__(self, width=0, height=0, x=0, y=0, ver_dist=0,
                 hor_dist=0, font_path=None, text_color=None):

        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.ver_dist = ver_dist
        self.hor_dist = hor_dist
        self.font_path = font_path
        self.text_color = text_color


class BoundingBox:
    """
        This class contains coordinates that represent text line bounding box
    """
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
