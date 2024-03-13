from aug2coco.settings import Settings
from aug2coco.data_process import _writeJson, load_check_json, _dfSplit
from aug2coco.json_process import _fillJson, set_constant
from aug2coco.schemes import PrevDataInfoHolder
import time

class Loader:
    """
    Loader class to perform loading, splitting and writing data
    """

    SETTINGS: Settings = None
    IDX: int = 0

    def __init__(self, settings: Settings):
        self.SETTINGS = settings
        self.IDX = 0

    def load(self, df):
        """
        :param df: dataframe of new data
        :return: -
        """
        print(f"Shape of total {df.shape}")
        holder = PrevDataInfoHolder()
        holder.INFO, holder.CATEGORIES_NEW, holder.CATEGORIES_TOTAL = set_constant(self.SETTINGS.UPLOAD,
                                                                                   self.SETTINGS.DATASET_DIR,
                                                                                   df)
        for i, split_type in enumerate(self.SETTINGS.SPLIT_TYPES):
            last_batch = False if i + 1 < len(self.SETTINGS.SPLIT_TYPES) else True
            type_df, self.IDX = _dfSplit(df, self.SETTINGS.SPLIT_RATE.get(split_type),
                                         self.IDX, last_batch)

            print(f"Shape of {split_type} {type_df.shape}")

            self.loadSplit(type_df, split_type, holder)
            holder.set_changes()

    def loadSplit(self, df, split_type, holder):
        """
        :param df: dataframe for plit
        :param split_type: train/val/test
        :param holder: holder od constant/previous data and indexes for this split
        :return: -
        """
        df, holder = load_check_json(self.SETTINGS.DATASET_DIR, split_type, self.SETTINGS.UPLOAD, df, holder)
        data = _fillJson(self.SETTINGS, df, holder, split_type=split_type)
        _writeJson(self.SETTINGS.DATASET_DIR, split_type, data, self.SETTINGS.UPLOAD)


