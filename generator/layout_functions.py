"""
    This module contains functions that are used to generate layouts for brochures
"""

import random
from random import randrange
import copy
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from generator import brochure as br, graphs as g, tables as t, text_functions as tf
import time

# -*- coding: utf-8 -*-

def generate_brochure(index, conf):
    """
        This function is used to put together brochure
    """

    text = tf.get_entire_text(index, conf.SAMPLES_PATH)
    brochure = generate_brochure_params(conf.SAMPLES_PATH)
    brochure.set_table_coords_arr()

    if random.choice([True, False]):

        header_bl = generate_header_params(brochure, [0.8, 1], [0.1, 0.15], [0.07, 0.1], [0.01, 1])
        header_bl.text_color = tf.get_contrasting_color(brochure.image, header_bl)
        header_bl.font_path = tf.get_random_file_path(os.path.join(conf.SAMPLES_PATH, 'sample_fonts'), '*.ttf')
        header, main_text = tf.extract_header(text, random.randint(len(text) // 200, len(text) // 30))
        put_text_in_block(header, brochure, header_bl, True)
        header_height = header_bl.height + int(header_bl.height * random.uniform(0.3, 0.5))
        segments = tf.divide_text(main_text, brochure.text_parts)

    else:

        segments = tf.divide_text(text, brochure.text_parts)
        header_height = 0

    brochure.header_height = header_height
    bl = generate_block_params(brochure)

    text_pos = 0
    for j in range(brochure.columns):
        for i in range(brochure.rows):

            choice = brochure.layout_matrix[i][j]
            bl.x = int(brochure.left_margin + j * (bl.width + bl.hor_dist))
            bl.y = int(brochure.top_margin + header_height + i * (bl.height + bl.ver_dist))
            bl.text_color = tf.get_contrasting_color(brochure.image, bl)
            bl.font_path = tf.get_random_file_path(os.path.join(conf.SAMPLES_PATH, 'sample_fonts'), '*.ttf')

            if choice == 1:
                determine_type_of_image(brochure, bl, index, conf, def_table=True)

            elif choice == 2:
                determine_type_of_image(brochure, bl, index, conf)

            elif choice == 3:
                text = segments[text_pos]
                text_pos += 1
                image_path = tf.get_random_file_path(os.path.join(conf.SAMPLES_PATH, 'sample_insert_images'), '*.jpg')
                put_text_and_image_in_block(text, brochure, bl, image_path)

            elif choice == 4:
                text = segments[text_pos] + segments[text_pos + 1]
                text_pos += 2
                header_bl = generate_header_params_for_block(bl, [0.8, 1], [0.17, 0.2],
                                                             [0.1, 0.12], [0.01, 1])
                header, main_text = tf.extract_header(text, random.randint(len(text) // 30,
                                                                           len(text) // 20))
                header_bl.font_path = bl.font_path
                header_bl.text_color = bl.text_color
                put_text_in_block(header, brochure, header_bl, True)

                block_header_height = header_bl.height + int(header_bl.height
                                                             * random.uniform(0.3, 0.5))
                text_bl = copy.deepcopy(bl)
                text_bl.y += block_header_height
                text_bl.height -= block_header_height
                put_text_in_block(main_text, brochure, text_bl, True)

            elif choice == 5:
                text = segments[text_pos] + segments[text_pos + 1]
                text_pos += 2
                put_text_in_block(text, brochure, bl, True)

    if conf.DICT_OUTPUT:
        return {"image": np.array(brochure.image), "boxes": brochure.t_cords, "image_name": f'{index}.jpg'}
    else:
        brochure.image.save(f"{os.path.join(conf.DATASET_PATH, 'images')}/{index}.jpg")


def generate_brochure_layout(width, height):
    """
        This function generates general brochure layout as a matrix and
        calculates in how many blocks you have to divide initial text
    """

    columns = random.randint(1, 3) if width > height else random.randint(1, 2)
    rows = random.randint(1, 2) if width > height else random.randint(1, 3)

    if rows == 1 and columns == 1:
        if width > height:
            columns = random.randint(2, 3)
        else:
            rows = random.randint(2, 3)

    matrix = [[0] * columns for _ in range(rows)]
    text_blocks_amount = 0
    tables_amount = 0

    for j in range(columns):
        for i in range(rows):

            choice = np.random.choice(np.arange(1, 6), p=[0.2, 0.2, 0.1, 0.2, 0.3])
            matrix[i][j] = choice
            if choice in (4, 5):
                text_blocks_amount += 2
            elif choice == 3:
                text_blocks_amount += 1
            elif choice == 1:
                tables_amount += 1

    if text_blocks_amount == 0:
        return generate_brochure_layout(width, height)

    if tables_amount == 0:
        j = randrange(columns)
        i = randrange(rows)
        matrix[i][j] = 1

    return matrix, text_blocks_amount, rows, columns


def determine_type_of_image(brochure, block, index, conf, def_table=False):
    """
        This function is used to decide on type of image to use for
        the given block and insert it
    """

    image_choice = np.random.choice(np.arange(1, 4), p=[0.1, 0.1, 0.8])

    if def_table or image_choice == 1:
        if conf.DICT_OUTPUT:
            tf.prepare_data_for_json(brochure, block, index, dict_output=True)
        else:
            tf.prepare_data_for_json(brochure, block, index,
                                     f"{os.path.join(conf.DATASET_PATH, 'table_locations')}/tables_{index}",
                                     dict_output=False)
        t.generate_table(brochure, block, conf)
        insert_path = conf.TEMPORARY_TABLE

    elif image_choice == 2:
        g.generate_plot(brochure, block, conf)
        insert_path = conf.TEMPORARY_IMAGE

    else:
        os.path.join(conf.SAMPLES_PATH, 'sample_insert_images')
        insert_path = tf.get_random_file_path(os.path.join(conf.SAMPLES_PATH, 'sample_insert_images'),'*.jpg')

    insert_image(brochure, insert_path, block)


def generate_brochure_params(samples_path):
    """
        This function is used to generate brochure parameters
    """

    path = tf.get_random_file_path(os.path.join(samples_path, 'sample_background_images'), '*.jpg')

    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    left_margin = random.randint(max(20, width // 20), width // 20)
    top_margin = random.randint(max(20, height // 20), height // 10)
    matrix, text_parts, rows, columns = generate_brochure_layout(width, height)
    brochure = br.Brochure(rows=rows, columns=columns, width=width,
                           height=height, layout_matrix=matrix,
                           text_parts=text_parts, left_margin=left_margin,
                           top_margin=top_margin, image=image, draw=draw)

    return brochure


def put_text_and_image_in_block(text, brochure, block, image_path):
    """
        This function scales image and text so that it fits in a
        given bounding box
    """

    is_vertical = block.height > block.width
    direction = random.choice(['down', 'up']) if is_vertical else random.choice(['left', 'right'])
    image_ratio = random.uniform(0.3, 0.5)
    image_size = int((block.height if is_vertical else block.width) * image_ratio)

    if is_vertical:

        image_margin = block.height * random.uniform(0.05, 0.1)
        x_image = block.x
        y_image = block.y + (block.height - image_size) if direction == 'down' else block.y
        x_text = block.x
        y_text = block.y if direction == 'down' else block.y + image_size + image_margin
        im_width = block.width
        im_height = image_size
        text_width = block.width
        text_height = block.height - image_size - image_margin

    else:

        image_margin = block.width * random.uniform(0.05, 0.1)
        x_image = block.x + (block.width - image_size) if direction == 'right' else block.x
        y_image = block.y
        x_text = block.x if direction == 'right' else block.x + image_size + image_margin
        y_text = block.y
        im_width = image_size
        im_height = block.height
        text_width = block.width - image_size - image_margin
        text_height = block.height

    image_block = br.BrochureBlock(width=im_width, height=im_height, x=x_image, y=y_image)
    text_block = br.BrochureBlock(width=text_width, height=text_height, x=x_text, y=y_text,
                                  font_path=block.font_path, text_color=block.text_color)

    insert_image(brochure, image_path, image_block)
    put_text_in_block(text, brochure, text_block, True)


def put_text_in_block(text, brochure, block, write_into_json, max_rows=0, font_size=0):
    """
        This function divides text in lines and scales them so that
        they fit in a given bounding box
    """

    if font_size == 0:
        font_size, lines = tf.scale_font(text, block.font_path, block.width, block.height, max_rows)
    else:
        lines = max_rows
    wrapped_text = tf.wrap_into_lines(text, lines)
    font = ImageFont.truetype(block.font_path, font_size)
    text_bbox = font.getbbox(text)
    text_height = text_bbox[3] - text_bbox[1]

    for i, line in enumerate(wrapped_text):

        y_final = block.y + i * text_height
        brochure.draw.text((block.x, y_final), line, font=font, fill=block.text_color)

        if write_into_json:
            project_text_line(block, y_final, line, font)


def project_text_line(block, y, line, font):
    """
        This function is used to project every text line onto white background
        for easy bounding box extraction
    """

    background_color = (255, 255, 255)
    text_color = (0, 0, 0)

    image = Image.new("RGB", (int(block.x + block.width + 50),
                              int(block.y + block.height + 50)), background_color)
    draw = ImageDraw.Draw(image)
    draw.text((block.x, y), line, font=font, fill=text_color)
    image_array = np.array(image)
    matching_pixels = np.argwhere(np.all(image_array < 200, axis=-1))
    margin = 1

    if len(matching_pixels) > 0:
        min_coordinates = np.min(matching_pixels, axis=0)
        max_coordinates = np.max(matching_pixels, axis=0)
        x1, y1 = int(min_coordinates[1]) - margin, int(min_coordinates[0]) - margin
        x2, y2 = int(max_coordinates[1]) + margin, int(max_coordinates[0]) + margin
        b_box = br.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    else:
        print("Bounding box not found in the image.")


def insert_image(brochure, path, block):
    """
        This inserts image that has given size at a given position
    """

    insert = Image.open(path)
    insert = insert.resize((block.width, block.height), Image.Resampling.LANCZOS)
    brochure.image.paste(insert, (block.x, block.y))


def generate_block_params(brochure):
    """
        This function generates margins for blocks of the brochure
    """

    remaining_height = brochure.height - 2 * brochure.top_margin - brochure.header_height
    remaining_width = brochure.width - 2 * brochure.left_margin
    dist_ver = int(remaining_height * random.uniform(0.02, 0.04))
    dist_hor = int(remaining_width * random.uniform(0.05, 0.07))
    width = (brochure.width - 2 * brochure.left_margin - (brochure.columns - 1)
             * dist_hor) // brochure.columns
    height = (remaining_height - brochure.rows * dist_ver) // brochure.rows

    block = br.BrochureBlock(width=width, height=height, ver_dist=dist_ver, hor_dist=dist_hor)
    return block


def generate_header_params(brochure, width_range, w_height_range, h_height_range, centring_range):
    """
        This function generates size parameters for headers
    """

    w = brochure.width - 2 * brochure.left_margin
    h = brochure.height - 2 * brochure.top_margin
    block = br.BrochureBlock(width=w, height=h, x=brochure.left_margin, y=brochure.top_margin)

    return generate_header_params_for_block(block, width_range, w_height_range,
                                            h_height_range, centring_range)


def generate_header_params_for_block(block, width_range, w_height_range,
                                     h_height_range, centring_range):
    """
        This function generates size parameters for headers
    """

    header_height_ratio = random.uniform(w_height_range[0], w_height_range[1]) \
        if block.width > block.height else random.uniform(h_height_range[0], h_height_range[1])
    header_height = int(block.height * header_height_ratio)
    header_width = int(block.width * random.uniform(width_range[0], width_range[1]))
    centre = (block.width - header_width) * random.uniform(centring_range[0], centring_range[1])
    header_block = br.BrochureBlock(width=header_width, height=header_height,
                                    x=block.x + centre, y=block.y)

    return header_block


def create_directory_if_not_exists(directory):
    """
    Check if a directory exists, and if not, create it.

    Args:
        directory (str): The directory path to check/create.

    Returns:
        bool: True if the directory exists or was successfully created, False otherwise.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
            return True
        except OSError as e:
            print(f"Failed to create directory '{directory}': {e}")
            return False
    else:
        print(f"Directory '{directory}' already exists.")
        return True
