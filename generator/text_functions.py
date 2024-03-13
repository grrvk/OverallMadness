"""
    This module contains functions that are used to work with text
"""

import random
import glob
import json
import csv
import os
from PIL import ImageFont
import numpy as np
from generator import brochure as br


# -*- coding: utf-8 -*-


def get_entire_text(index, samples_path):
    """
        This function is used to get text that is used to
        generate brochure from random text file
    """

    general_path = os.path.join(samples_path, 'sample_text')
    text_files = glob.glob(os.path.join(general_path, '*.txt'))
    text_file = random.choice(text_files)

    with open(text_file, 'r', encoding='utf-8') as f:
        entire_text = f.read()

    return entire_text


def get_str_from_text(file):
    lines = open(file).readlines()
    line = lines[random.randrange(len(lines))]
    return line


def write_to_csv(image_name, text_file_name, path_to_csv):
    """
        This function is used for writing into csv file
    """

    data = {'image_name': image_name, 'text_file_name': text_file_name}
    file_exists = os.path.isfile(path_to_csv)

    with open(path_to_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'text_file_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        found = False
        with open(path_to_csv, 'r', newline='', encoding='utf-8') as readfile:
            reader = csv.DictReader(readfile)
            for row in reader:
                if row['image_name'] == image_name:
                    found = True
                    break
        if found:
            rows = []
            with open(path_to_csv, 'r', newline='', encoding='utf-8') as readfile:
                reader = csv.DictReader(readfile)
                for row in reader:
                    if row['image_name'] == image_name:
                        rows.append(data)
                    else:
                        rows.append(row)

            with open(path_to_csv, 'w', newline='', encoding='utf-8') as writefile:
                writer = csv.DictWriter(writefile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            writer.writerow(data)


def get_random_color(min_val, max_val):
    """
        This function returns red, green and blue values of a color that are
        randomly generated in a given range
    """

    red = random.randint(min_val, max_val)
    green = random.randint(min_val, max_val)
    blue = random.randint(min_val, max_val)

    return red, green, blue


def get_contrasting_color(image=None, block=None, color=None):
    """
        This module generates color that is contrasting to the average color
        of the given image
    """

    if color is not None:
        average_color = color
    else:
        cropped = image.crop((block.x, block.y, block.x + block.width, block.y + block.height))
        average_color_row = np.average(cropped, axis=0)
        average_color = np.average(average_color_row, axis=0)

    average_color = tuple(int(value) for value in average_color)
    result = []

    for i in range(3):
        if average_color[i] < 128:
            result.append(random.randint(128 + average_color[i], 255))
        else:
            result.append(random.randint(0, average_color[i] - 128))

    return tuple(result)


def get_random_file_path(directory, file_type):
    """
        This function returns path of a randon file of the given type
    """

    files = glob.glob(os.path.join(directory, file_type))
    file = random.choice(files)
    return file


def scale_font(txt, font_path, width, height, max_rows=0):
    """
        This function scales size of a given text so that it fits in the
        given bounding box and returns font size and number of lines in which text
        should be divided
    """

    font_size = 1
    rows = 1

    while True:

        font = ImageFont.truetype(font_path, font_size)
        text_bbox = font.getbbox(txt)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if text_width / rows < width * 0.96 and text_height * rows < height:
            font_size += 1
            continue

        if 0 < max_rows == rows:
            break

        if text_height * (rows + 1) < height:
            rows += 1
            continue

        break

    return font_size - 1, rows


def divide_text(text, number_of_parts):
    """
        This function divides text in given number of parts, so that each part
        contains full words
    """

    words = text.split()
    words_per_part = len(words) // number_of_parts
    parts = []
    current_part = []
    current_word_count = 0

    for word in words:

        current_part.append(word)
        current_word_count += 1

        if current_word_count >= words_per_part:
            parts.append(' '.join(current_part))
            current_part = []
            current_word_count = 0

    if len(current_part) > 0:
        last_part = parts[number_of_parts - 1]
        last_part += ' '.join(current_part)
        parts[number_of_parts - 1] = last_part

    return parts


def extract_header(text, min_header_length):
    """
        This function extracts header so that it consists only of full words
        and is longer than the minimum given length
    """

    words = text.split()
    selected_words = []
    remaining_text = text

    for word in words:
        selected_words.append(word)
        if len(' '.join(selected_words)) >= min_header_length:
            remaining_text = ' '.join(words[len(selected_words):])
            break

    header = ' '.join(selected_words)
    return header, remaining_text


def wrap_into_lines(text, lines):
    """
        This function wraps text into given number of lines, so that each line is exactly the
        same characters long with respect to the remainder
    """

    segments = []
    text_length = len(text)
    segment_length = text_length // lines
    remainder = text_length % lines
    start = 0

    for _ in range(0, lines):
        end = start + segment_length + (1 if remainder > 0 else 0)
        segments.append(text[start:end])
        start = end
        remainder -= 1

    return segments


def prepare_data_for_json(brochure, block, index, json_file_path: str = None, dict_output: bool=False, table=False):
    """
        This function transforms block data into bounding box object
    """

    x1, y1 = block.x, block.y
    x2, y2 = block.x + block.width, block.y + block.height
    b_box = br.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    if dict_output:
        append_bbox_cords(brochure, b_box)
    else:
        append_to_json(json_file_path, b_box, index, table)

def append_bbox_cords(brochure, b_box):
    """
        This function is used to append text line coordinates to brochure t_cords field
    """

    new_entry = {
        "x1": b_box.x1,
        "y1": b_box.y1,
        "x2": b_box.x2,
        "y2": b_box.y2
    }

    brochure.add_coords(new_entry)


def append_to_json(file_path, b_box, index, table=False):
    """
        This function is used to append text line coordinates to json file
    """

    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            existing_data = json.load(json_file)
    except (json.JSONDecodeError, FileNotFoundError):
        existing_data = []

    if table:
        new_entry = {
            "image": str(index) + ".jpg",
            "category": "Table",
            "x1": b_box.x1,
            "y1": b_box.y1,
            "x2": b_box.x2,
            "y2": b_box.y2
        }
    else:
        new_entry = {
            "x1": b_box.x1,
            "y1": b_box.y1,
            "x2": b_box.x2,
            "y2": b_box.y2
        }

    existing_data.append(new_entry)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, indent=4)


def delete_file(file_path):
    """
        This function is used to automatically delete json file
    """

    try:
        os.remove(file_path)
        print(f"The file {file_path} has been deleted.")
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")


def remove_tables_from_json(file_path, image_name):
    """
        This function is used to remove all tables that have been generated
        for image that was deleted
    """

    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            existing_data = json.load(json_file)
    except (json.JSONDecodeError, FileNotFoundError):
        existing_data = []

    tables = [entry for entry in existing_data if entry.get("image") != image_name]
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(tables, json_file, indent=4)
