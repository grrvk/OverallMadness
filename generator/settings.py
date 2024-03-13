import os
import shutil
import glob
from enum import Enum
import PIL


class Types(str, Enum):
    images = 'jpg'
    fonts = 'ttf'
    text = 'txt'


def _mkdir(name):
    """
        :param name: path to directory to clear and create
        :return: None
    """
    if os.path.exists(name) is False:
        try:
            os.mkdir(name)
        except Exception as e:
            print('Failed to create directory. Reason: %s' % e)
    else:
        for filename in os.listdir(name):
            file_path = os.path.join(name, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def create_directories(dataset_path: str, dir_names: list):
    for split_type in dir_names:
        _mkdir(os.path.join(dataset_path, split_type))


def create_temporary_picture(samples_path: str, name: str):
    temporary_picture = PIL.Image.new(mode="RGB", size=(200, 200), color=(0, 0, 0))
    temporary_picture.save(os.path.join(samples_path, f'{name}.jpg'))


def remove_temporary_picture(samples_path: str, name: str):
    os.remove(f'{os.path.join(samples_path, name)}.jpg')


def get_number_of_files_by_type(path, dir_name):
    dir_name_end = dir_name.split('_')[-1]
    return len(glob.glob1(os.path.join(path, dir_name), f"*.{Types[dir_name_end].value}"))


def check_files_size(path, name):
    file = os.path.join(path, name)
    if not os.path.exists(file) or os.path.getsize(file) == 0:
        raise Exception(f'File {name} must exist and be not empty')


class Settings:
    DATASET_PATH: str = None
    SAMPLES_PATH: str = None
    NAMING_INDEX: int
    UPLOAD: bool

    DATASET_IMAGES_PATH: str = None
    DATASET_TABLE_LOCATIONS_PATH: str = None

    TEMPORARY_IMAGE: str
    TEMPORARY_TABLE: str

    NEC_SAMPLES_DIRS: list = ['sample_background_images', 'sample_fonts', 'sample_insert_images',
                              'sample_table_fonts', 'sample_text']
    NEC_DATASET_DIRS: list = ['images', 'table_locations']
    NEC_DATASET_FILES: list = ['content_sentences', 'content_words']
    DICT_OUTPUT : bool = False

    def __repr__(self):
        return (f"Config: dataset_path = {self.DATASET_PATH}, \n"
                f"samples_path = {self.SAMPLES_PATH}, \n"
                f"indexing_increment = {self.NAMING_INDEX}, \n"
                f"is_uploading = {self.UPLOAD} \n")

    def __init__(self, **kwargs):
        self.UPLOAD = kwargs.get('upload')
        samples_path = kwargs.get('samples_path')
        if samples_path is None or not os.path.exists(samples_path):
            raise Exception('Path to samples does not exist')
        self.SAMPLES_PATH = samples_path
        self.check_samples(samples_path)

        self.TEMPORARY_IMAGE = os.path.join(self.SAMPLES_PATH, 'temp_img.jpg')
        self.TEMPORARY_TABLE = os.path.join(self.SAMPLES_PATH, 'temp_img.jpg')

        if kwargs.get("dict_output"):
            self.DICT_OUTPUT = True
        else:
            self.folder_output(**kwargs)

    def folder_output(self, **kwargs):
        dataset_path = kwargs.get('dataset_path')
        dataset_name = kwargs.get('dataset_name')

        if self.UPLOAD:
            self.load(dataset_path)
        else:
            self.create(dataset_path, dataset_name)

    def load(self, dataset_path):
        print('Uploading to existing dataset')
        if dataset_path is None or not os.path.exists(dataset_path):
            raise Exception('Path to where to create dataset does not exist')

        self.DATASET_PATH = dataset_path
        self.NAMING_INDEX = len(os.listdir(os.path.join(self.DATASET_PATH, 'images')))
        dir_list = os.listdir(self.DATASET_PATH)
        dirs_to_create = [n for n in self.NEC_DATASET_DIRS if not (n in dir_list)]
        create_directories(self.DATASET_PATH, dirs_to_create)
        self.NAMING_INDEX = len(os.listdir(os.path.join(self.DATASET_PATH, 'images')))

    def create(self, dataset_path, name):
        print('Creating dataset')
        dataset_name = name if name is not None else 'dataset'

        if dataset_path is None:
            self.DATASET_PATH = dataset_name
        elif not os.path.exists(dataset_path):
            raise Exception('Path to where to create dataset does not exist')
        else:
            self.DATASET_PATH = os.path.join(dataset_path, dataset_name)
        _mkdir(self.DATASET_PATH)
        create_directories(self.DATASET_PATH, self.NEC_DATASET_DIRS)
        self.NAMING_INDEX = 0

    def check_samples(self, path):
        samples_dirs_list = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        for dir_name in self.NEC_SAMPLES_DIRS:
            if (not (dir_name in samples_dirs_list)) or get_number_of_files_by_type(path, dir_name) == 0:
                raise Exception(f'Samples folder must have {dir_name} folder present and not empty')
        for file_name in self.NEC_DATASET_FILES:
            check_files_size(self.SAMPLES_PATH, file_name)
        create_temporary_picture(path, 'temp_img')

    def clear_temp_samples(self):
        remove_temporary_picture(self.SAMPLES_PATH, 'temp_img')


def setGenSettings(samples_path: str, dataset_path: str = None, dataset_name: str = None,
                upload: bool = False, dict_output: bool = False):

    return Settings(samples_path=samples_path,
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    upload=upload,
                    dict_output=dict_output)
