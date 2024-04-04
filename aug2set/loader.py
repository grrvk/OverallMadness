import numpy as np
from PIL import Image
from aug2set.settings import Settings
from aug2set.data_process import _writeJson, load_check_json, _dfSplit, draw_mask, save_semantic
from aug2set.json_process import _fillJson, set_constant
from aug2set.schemes import PrevDataInfoHolder


class Loader:
    """
    Loader class to perform loading, splitting and writing data
    """

    SETTINGS: Settings = None
    IDX: int = 0
    DATASET_TYPE: list = ['coco', 'semantic']

    def __init__(self, settings: Settings):
        self.SETTINGS = settings
        self.IDX = 0

    def load(self, df):
        """
        :param df: dataframe of new data
        :return: -
        """
        print(f"Shape of total {df.shape}")
        if self.SETTINGS.DATASET_TYPE == 'coco':
            holder = PrevDataInfoHolder()
            holder.INFO, holder.CATEGORIES_NEW, holder.CATEGORIES_TOTAL = set_constant(self.SETTINGS.UPLOAD,
                                                                                       self.SETTINGS.DATASET_DIR,
                                                                                       df)
            for i, split_type in enumerate(self.SETTINGS.SPLIT_TYPES):
                last_batch = False if i + 1 < len(self.SETTINGS.SPLIT_TYPES) else True
                type_df, self.IDX = _dfSplit(df, self.SETTINGS.SPLIT_RATE.get(split_type),
                                             self.IDX, last_batch)

                print(f"Shape of {split_type} {type_df.shape}")

                self.loadSplitCoco(type_df, split_type, holder)
                holder.set_changes()
        else:
            for i, split_type in enumerate(self.SETTINGS.SPLIT_TYPES):
                last_batch = False if i + 1 < len(self.SETTINGS.SPLIT_TYPES) else True
                type_df, self.IDX = _dfSplit(df, self.SETTINGS.SPLIT_RATE.get(split_type),
                                             self.IDX, last_batch)

                print(f"Shape of {split_type} {type_df.shape}")

                self.loadSplitSem(type_df, split_type)

    def loadSplitCoco(self, df, split_type, holder):
        """
        :param df: dataframe for plit
        :param split_type: train/val/test
        :param holder: holder od constant/previous data and indexes for this split
        :return: -
        """
        df, holder = load_check_json(self.SETTINGS.DATASET_DIR, split_type, self.SETTINGS.UPLOAD, df, holder)
        data = _fillJson(self.SETTINGS, df, holder, split_type=split_type)
        _writeJson(self.SETTINGS.DATASET_DIR, split_type, data, self.SETTINGS.UPLOAD)

    def loadSplitSem(self, df, split_type):
        image_paths = list(set(df['image'].tolist()))
        for image_path in image_paths:
            df_for_image = df[df['image'] == image_path]
            image = df_for_image['PIL'][0]
            #mask = np.zeros(df_for_image['PIL'].values[0].shape)

            class_map = np.zeros_like(image[:, :, 0], dtype=np.uint8)
            instance_map = np.zeros_like(image[:, :, 0], dtype=np.uint8)

            for index, row in df_for_image.iterrows():
                #print(image.shape)
                mask = Image.new("L", (image.shape[1], image.shape[0]), 0)

                mask = draw_mask(mask, row['segmentation'][0])
                instance_id = index + 1

                class_map[mask > 0] = 1
                instance_map[mask > 0] = instance_id

            print(np.unique(class_map))
            print(np.unique(instance_map))

            mask_image = np.zeros_like(image, dtype=np.uint8)
            mask_image[:, :, 0] = class_map
            mask_image[:, :, 1] = instance_map

            print(np.unique(mask_image[:, :, 0]))
            print(np.unique(mask_image[:, :, 1]))

            save_semantic(self.SETTINGS.DATASET_DIR, split_type, image_path, image, mask_image)
