"""
    This module is used to generate tables for brochures
"""

import os
import copy
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from generator import brochure as br, layout_functions as lf, text_functions as tf
# -*- coding: utf-8 -*-


class Table:
    """
        This class is used to represent table structure
    """

    def __init__(self, rows, columns, width, height, cell_width, cell_height,
                 cell_left_margin, cell_top_margin, draw, image, font_type, font_color,
                 background_color, words_path, sentences_path):

        self.rows = rows
        self.columns = columns
        self.width = width
        self.height = height
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.cell_left_margin = cell_left_margin
        self.cell_top_margin = cell_top_margin
        self.draw = draw
        self.image = image
        self.font_type_path = font_type
        self.font_color = font_color
        self.background_color = background_color
        self.words_path = words_path
        self.sentences_path = sentences_path


def decide_on_coloring_type(brochure, block):
    """
        This function is used to randomly decide on coloring type
        for a table
    """

    coloring = np.random.choice(np.arange(1, 4), p=[0.6, 0.2, 0.2])

    if coloring == 1:

        font_color = (0, 0, 0)
        background_color = (255, 255, 255)

    elif coloring == 2:

        cropped = brochure.image.crop((block.x, block.y, block.x + block.width,
                                       block.y + block.height))
        average_color_row = np.average(cropped, axis=0)
        background_color = np.average(average_color_row, axis=0)
        background_color = tuple(int(value) for value in background_color)
        font_color = tf.get_contrasting_color(color=background_color)

    elif coloring == 3:

        font_color = tf.get_random_color(0, 255)
        background_color = tf.get_contrasting_color(color=font_color)

    return font_color, background_color


def draw_table_borders(table):
    """
        This function is used to draw a table layout
    """

    row_lines = random.choice([True, False])
    col_lines = random.choice([True, False])
    borders = random.choice([True, False])
    inner_borders = random.choice([True, False])

    if row_lines:
        for i in range(table.rows - 1):
            draw_line(table.draw, (0, (i + 1) * table.cell_height),
                      (table.width, (i + 1) * table.cell_height), table.font_color)

    if col_lines:
        for j in range(table.columns - 1):
            draw_line(table.draw, ((j + 1) * table.cell_width, 0),
                      ((j + 1) * table.cell_width, table.height), table.font_color)

    if borders:
        draw_rectangle(table.draw, (0, 0), (table.width - 1, table.height - 1), table.font_color, 2)

    if inner_borders:
        draw_line(table.draw, (0, table.cell_height), (table.width, table.cell_height),
                  table.font_color, 2)
        draw_line(table.draw, (table.cell_width, 0), (table.cell_width, table.height),
                  table.font_color, 2)


def draw_line(draw, start, end, color, line_width=1):
    """
        This function is used to draw a line to create layout for the table
    """

    shape = [start, end]
    draw.line(shape, fill=color, width=line_width)


def draw_rectangle(draw, start, end, color, outline_width=1):
    """
        This function is used to draw a rectangle to create layout for the table
    """

    shape = [start, end]
    draw.rectangle(shape, outline=color, width=outline_width)


def generate_and_scale_table_content(table):
    """
        This function is used to scale text in all cels of the regular table
        to the same size
    """

    matrix_content = [[0 for _ in range(table.columns)] for _ in range(table.rows)]
    min_font = 10000
    text_width = table.cell_width - 2 * table.cell_left_margin
    text_height = table.cell_height - 2 * table.cell_top_margin

    for j in range(table.columns):
        numbers = random.choice([True, False])
        digits = random.randint(0, 3)
        for i in range(table.rows):
            if numbers and j and i:
                matrix_content[i][j] = str(round(random.uniform(-100, 100), digits))
            else:
                matrix_content[i][j] = tf.get_str_from_text(table.words_path)

            font_size, _ = tf.scale_font(matrix_content[i][j], table.font_type_path, text_width,
                                         text_height, 1)
            if font_size < min_font:
                min_font = font_size

    return matrix_content, min_font


def generate_table_params(brochure, block, conf):
    """
        This function is used to generate instance of table class that
        represents table
    """

    columns = random.randint(block.width // 100, block.width // 80)
    rows = random.randint(block.height // 35, block.height // 28)
    columns = max(columns, 2)
    rows = max(rows, 2)

    cell_width = block.width // columns
    cell_height = block.height // rows
    left_margin = int(cell_width * 0.1)
    top_margin = int(cell_height * 0.1)

    table_font_path = tf.get_random_file_path(os.path.join(conf.SAMPLES_PATH, 'sample_table_fonts'), '*ttf')
    words_path = os.path.join(conf.SAMPLES_PATH, 'content_words.txt')
    sentences_path = os.path.join(conf.SAMPLES_PATH, 'content_sentences.txt')

    font_color, background_color = decide_on_coloring_type(brochure, block)
    image = Image.new('RGB', (block.width, block.height), background_color)
    draw = ImageDraw.Draw(image)

    table = Table(rows=rows, columns=columns, width=block.width, height=block.height,
                  cell_width=cell_width, cell_height=cell_height,
                  cell_left_margin=left_margin, cell_top_margin=top_margin,
                  draw=draw, image=image, font_type=table_font_path, font_color=font_color,
                  background_color=background_color, words_path=words_path, sentences_path=sentences_path)

    return table


def get_irregular_horizontal_layout(remaining_space, horizontal_widths):
    """
        This function is used to draw a table layout
    """

    i = 0
    while remaining_space > 0:

        if i == 0 and horizontal_widths[i] == 1:
            horizontal_widths[i] += 1
            remaining_space -= 1
            continue

        if random.choice([True, False]):
            horizontal_widths[i] += 1
            remaining_space -= 1
        i += 1

        if i == len(horizontal_widths):
            i = 0

    return horizontal_widths


def generate_cell_content(vertical_width, horizontal_width, table):
    """
        This function is used to randomly decide on text content for
        table cell
    """

    if vertical_width + horizontal_width > 3:
        return tf.get_str_from_text(table.sentences_path)

    if random.choice([True, False]):
        digits = random.randint(0, 5)
        return str(round(random.uniform(-100, 100), digits))

    return tf.get_str_from_text(table.words_path)


def draw_irregular_layout(table, horizontal_widths, index, vertical_widths):
    """
        This function is used to draw cell borders for irregular table
    """

    left_margin = 0
    for i in range(0, index):
        left_margin += vertical_widths[i]
    left_margin *= table.cell_width

    if horizontal_widths == 0:
        draw_line(table.draw, (left_margin, 0), (left_margin, table.height), table.font_color, 2)
        return

    column_width = vertical_widths[index] * table.cell_width
    for i in range(len(horizontal_widths)):

        draw_line(table.draw, (left_margin, 0), (left_margin, table.height), table.font_color, 2)
        top_margin = 0

        for j in range(len(horizontal_widths) - 1):
            top_margin += horizontal_widths[j] * table.cell_height
            draw_line(table.draw, (left_margin, top_margin),
                      (column_width + left_margin, top_margin), table.font_color)


def get_probabilities_for_layouts(horizontal_layout_types):
    """
        This function is used to get probabilities for generating different vertical
        layout types
    """

    possibility_of_zero = 0.1
    other_pos = (1 - possibility_of_zero) / (horizontal_layout_types - 1)
    return [possibility_of_zero] + [other_pos] * (horizontal_layout_types - 1)


def generate_irregular_table_layout(table):
    """
        This function is used to randomly generate structure for irregular table
    """

    max_val = max(2, table.columns)
    vertical_blocks = random.randint(max(2, table.columns // 2), max_val)
    vertical_widths = [1 for _ in range(vertical_blocks)]
    remaining_columns = table.columns - vertical_blocks
    horizontal_layout_types = random.randint(3, 5)

    i = 0
    while remaining_columns > 0:
        if random.choice([True, False]):
            vertical_widths[i] += 1
            remaining_columns -= 1
        i += 1
        if i == len(vertical_widths):
            i = 0

    layouts = [0]
    for i in range(0, horizontal_layout_types):
        max_val = max(2, table.rows)
        horizontal_blocks = random.randint(max(2, table.rows // 2), max_val)
        horizontal_widths = [1 for _ in range(horizontal_blocks)]
        remaining_rows = table.rows - horizontal_blocks
        layouts.append(get_irregular_horizontal_layout(copy.deepcopy(remaining_rows),
                                             copy.deepcopy(horizontal_widths)))

    p = get_probabilities_for_layouts(horizontal_layout_types)
    current_width = 0

    for i in range(vertical_blocks):
        layout = np.random.choice(np.arange(0, horizontal_layout_types), p=p)
        draw_irregular_layout(table, layouts[layout], i, vertical_widths)
        fill_irregular_table(table, layout, vertical_widths[i], current_width, layouts)
        current_width += vertical_widths[i] * table.cell_width

    draw_rectangle(table.draw, (0, 0), (table.width - 1, table.height - 1), table.font_color, 2)


def fill_irregular_table(table, column_layout_type, vertical_width, current_width, layouts):
    """
        This function is used to write text into each cell of irregular table
    """

    if column_layout_type == 0:
        text = generate_cell_content(vertical_width, 3, table)
        block = br.BrochureBlock(width=vertical_width * table.cell_width,
                                 height=table.height, x=current_width, y=0)

        scale_and_write_vertical_text(table, text, block, tf.get_str_from_text(table.sentences_path))
        return

    horizontal_widths = layouts[column_layout_type]
    divide_flag, divide_pos = get_divide_pos(vertical_width, horizontal_widths)
    top_margin = 0

    for i in range(divide_pos):
        block_width = vertical_width * table.cell_width
        block_height = horizontal_widths[i] * table.cell_height
        text_block = br.BrochureBlock(width=block_width, height=block_height,
                                      x=current_width, y=top_margin)
        text = generate_cell_content(vertical_width, horizontal_widths[i], table)

        if block_width > block_height:
            scale_and_write_horizontal_text(table, text, text_block, tf.get_str_from_text(table.sentences_path))
        else:
            scale_and_write_vertical_text(table, text, text_block, tf.get_str_from_text(table.sentences_path))
        top_margin += horizontal_widths[i] * table.cell_height

    if divide_flag:
        fill_divided_column(table, current_width, vertical_width, top_margin,
                            horizontal_widths, divide_pos)


def fill_divided_column(table, current_width, vertical_width, top_margin,
                        horizontal_widths, divide_pos):
    """
        This function is used to fill divided column of irregular table
        with random data
    """

    first_width = random.randint(1, vertical_width - 1)
    block1_width, block2_width = first_width * table.cell_width, \
        (vertical_width - first_width) * table.cell_width
    draw_line(table.draw, (current_width + block1_width, top_margin),
              (current_width + block1_width, table.height), table.font_color)

    for j in range(divide_pos, len(horizontal_widths)):

        block_height = horizontal_widths[j] * table.cell_height
        block_1 = br.BrochureBlock(width=block1_width, height=block_height,
                                   x=current_width, y=top_margin)
        block_2 = br.BrochureBlock(width=block2_width, height=block_height,
                                   x=current_width + block1_width, y=top_margin)

        text_1 = generate_cell_content(first_width, horizontal_widths[j], table)
        text_2 = generate_cell_content(vertical_width - first_width, horizontal_widths[j], table)

        if block1_width > block_height:
            scale_and_write_horizontal_text(table, text_1, block_1,  tf.get_str_from_text(table.sentences_path))
        else:
            scale_and_write_vertical_text(table, text_1, block_1,  tf.get_str_from_text(table.sentences_path))

        if block2_width > block_height:
            scale_and_write_horizontal_text(table, text_2, block_2, tf.get_str_from_text(table.sentences_path))
        else:
            scale_and_write_vertical_text(table, text_2, block_2, tf.get_str_from_text(table.sentences_path))

        top_margin += horizontal_widths[j] * table.cell_height


def get_divide_pos(vertical_width, horizontal_widths):
    """
        This function is used to randomly decide if the column will de divided
         and if so decide position at which column will be divided
    """

    divide_flag = vertical_width >= 2 and random.choice([True, False])
    divide_pos = random.randint(1, len(horizontal_widths) // 2) \
        if divide_flag else len(horizontal_widths)
    return divide_flag, divide_pos


def get_text_width_and_height(text, font_path, font_size, rows=1):
    """
        This function is used to get width and height of a text block
    """

    font = ImageFont.truetype(font_path, font_size)
    text_bbox = font.getbbox(text)
    return (text_bbox[2] - text_bbox[0]) / rows, (text_bbox[3] - text_bbox[1]) * rows


def scale_and_write_vertical_text(table, text, block, long_text):
    """
        This function is used to scale and write vertical text into the cell
        of the given irregular table
    """

    margin = 4
    text_width, text_height = get_suitable_text_size(text, block.height,
                                                     block.width, margin, long_text, table.font_type_path)

    cell = Image.new("RGB", (block.height - margin, block.width - margin), table.background_color)
    draw = ImageDraw.Draw(cell)
    background = br.Brochure(image=cell, draw=draw)

    height_coefficient = 0.3 * block.width / block.height
    x = int((block.height - text_width) // 2)
    y = int(height_coefficient * block.width)
    bl = br.BrochureBlock(width=text_width - margin, height=text_height, x=x, y=y,
                          font_path=table.font_type_path, text_color=table.font_color)

    lf.put_text_in_block(text, background, bl, False)
    cell = cell.rotate(90, expand=True)
    paste_position = (block.x + margin//2, block.y + margin//2)
    table.image.paste(cell, paste_position)


def scale_and_write_horizontal_text(table, text, block, long_text):
    """
        This function is used to scale and write horizontal text into the cell
        of the given irregular table
    """

    text_width, text_height = get_suitable_text_size(text, block.width, block.height, 0, long_text, table.font_type_path)

    height_coefficient = 0.3 * block.height / block.width
    x = int(block.x + (block.width - text_width) // 2)
    y = int(block.y + (height_coefficient * block.height))
    bl = br.BrochureBlock(width=text_width, height=text_height, x=x, y=y, font_path=table.font_type_path,
                          text_color=table.font_color)

    lf.put_text_in_block(text, table, bl, False)


def get_suitable_text_size(text, width, height, margin, long_text, font):
    """
        This function is used to get text size that will be suitable for
        certain irregular table cell
    """

    width -= margin
    height -= margin
    max_text_width = int(width * 0.8)
    max_text_height = int(height * 0.5)

    if long_text:
        font_size, r = tf.scale_font(text, font, max_text_width, max_text_height)
    else:
        font_size, r = tf.scale_font(text, font, max_text_width, max_text_height, 1)

    text_width, text_height = get_text_width_and_height(text, font, font_size, r)
    return text_width, text_height


def generate_table(brochure, block, conf):
    """
        This function is used to generate table for brochure
        based on it`s future width and height
    """

    table = generate_table_params(brochure, block, conf)
    content_matrix, font_size = generate_and_scale_table_content(table)

    if np.random.choice([True, False], p=[0.7, 0.3]):
        generate_irregular_table_layout(table)
    else:
        draw_table_borders(table)
        for j in range(table.columns):
            for i in range(table.rows):

                x = int(table.cell_left_margin + j * table.cell_width)
                y = int(table.cell_top_margin + i * table.cell_height)
                bl = br.BrochureBlock(width=table.cell_width - 2 * table.cell_left_margin,
                                  height=table.cell_height - 2 * table.cell_top_margin,
                                  x=x, y=y, font_path=table.font_type_path, text_color=table.font_color)

                lf.put_text_in_block(content_matrix[i][j], table, bl, False, 1, font_size)

    table.image.save(conf.TEMPORARY_TABLE)
