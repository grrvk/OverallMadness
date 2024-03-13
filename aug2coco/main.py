import pandas as pd
from aug2coco.settings import Settings
from aug2coco.data_process import _loadImages
from aug2coco.loader import Loader


def convert_withZip(settings: Settings):
    print(settings)

    loader = Loader(settings)
    df = _loadImages(settings.WORKING_DIR)
    loader.load(df)

    print(f"Dataset generated successfully!")


def convert_withDf(settings: Settings, df: pd.DataFrame):
    print(settings)

    loader = Loader(settings)
    loader.load(df)

    print(f"Dataset generated successfully!")
